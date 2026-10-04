import numpy as np
import cv2
import logging
from pipeline.Stage import Stage

logger = logging.getLogger(__name__)


# Constraints (hard filters) - disabled to allow all tracks through
ASPECT_RATIO_MIN = 0.0  # Allow any aspect ratio
ASPECT_RATIO_MAX = 10.0
BBOX_SIZE_MIN = 0.0     # Allow any size
BBOX_SIZE_MAX = 1.0

# Weights for the per-track score (easy to adjust)
WEIGHTS = {
    "confidence": 0.4,  # Higher detection confidence is better
    "thickness": 0.35,  # Chosen point sits on a thick part of the branch
    "center": 0.25,     # Chosen point is close to the image centre (less corrective motion)
}

# Which skeleton pixel of a track becomes its perch point
PIXEL_WEIGHTS = {"thickness": 0.6, "center": 0.4}
END_MARGIN_PX = 30         # keep the point this far from skeleton endpoints (gripper needs branch on both sides)
TRACK_SWITCH_MARGIN = 0.05 # a rival track must beat the previous best by this much to take over
POINT_KEEP_RADIUS = 60     # px, previous point follows the branch to the nearest valid skeleton pixel within this
POINT_KEEP_MARGIN = 0.3    # ... as long as it scores at most this much below the new best pixel
                           # (tuned on the recordings: 6 px / 0.03 made the point jump 30+ px between frames)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))


def _endpoints(sk):
    """Skeleton pixels with exactly one 8-neighbour. Returns [E, 2] (row, col)."""
    neighbours = cv2.filter2D(sk.astype(np.uint8), -1, np.ones((3, 3), np.uint8),
                              borderType=cv2.BORDER_CONSTANT)
    return np.argwhere(sk & (neighbours == 2))


class CandidateScoring(Stage):
    """Picks, for every tracked branch, one perch point ON its skeleton and ranks the tracks.

    Uses all post-processing outputs: the track's own mask limits where the point
    may be, the skeleton gives the branch axis, the distance heatmap gives the
    local thickness. Everything is argmax / tie-broken by pixel order, so the same
    input always gives the same point.
    """

    def __init__(self):
        self._prev_id = None
        self._prev_xy = {}  # track_id -> (x, y) chosen last frame

    def _pick_point(self, mask_i, xyxy, ctx, track_id, frame_w, frame_h):
        """Returns (x, y, heat_norm, center_norm, on_skeleton, relaxed) or None if no skeleton pixel."""
        if mask_i is None or ctx.skeleton is None or ctx.distance_heatmap is None:
            return None

        x1, y1, x2, y2 = xyxy
        x1, y1 = max(int(x1) - 1, 0), max(int(y1) - 1, 0)
        x2, y2 = min(int(x2) + 2, frame_w), min(int(y2) + 2, frame_h)
        if x2 <= x1 or y2 <= y1:
            return None

        sk = (ctx.skeleton[y1:y2, x1:x2] > 0) & mask_i[y1:y2, x1:x2]
        pts = np.argwhere(sk)  # row-major, so ties resolve to the lowest (y, x)
        if len(pts) == 0:
            return None

        # Local thickness, renormalised per track so other branches in view don't change this one's score
        heat = ctx.distance_heatmap[y1:y2, x1:x2][pts[:, 0], pts[:, 1]]
        heat_norm = heat / heat.max() if heat.max() > 0 else np.ones_like(heat)

        # Stay away from the ends of the skeleton, relax if that would leave nothing
        relaxed = False
        ends = _endpoints(sk)
        keep = np.ones(len(pts), dtype=bool)
        if len(ends) > 0:
            d_end = np.sqrt(((pts[:, None, :] - ends[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
            keep = d_end >= END_MARGIN_PX
            if not keep.any():
                keep = np.ones(len(pts), dtype=bool)
                relaxed = True
        pts, heat_norm = pts[keep], heat_norm[keep]

        xs = pts[:, 1] + x1
        ys = pts[:, 0] + y1
        max_dist = np.hypot(frame_w, frame_h) / 2
        center_norm = 1.0 - np.hypot(xs - frame_w / 2, ys - frame_h / 2) / max_dist
        pixel_score = PIXEL_WEIGHTS["thickness"] * heat_norm + PIXEL_WEIGHTS["center"] * center_norm

        best = int(np.argmax(pixel_score))

        # Keep last frame's point while it is still (almost) as good, so the point doesn't wander
        prev = self._prev_xy.get(track_id)
        if prev is not None:
            d_prev = np.hypot(xs - prev[0], ys - prev[1])
            near = np.where(d_prev <= POINT_KEEP_RADIUS)[0]
            if len(near) > 0:
                j = near[int(np.argmin(d_prev[near]))]
                if pixel_score[j] >= pixel_score[best] - POINT_KEEP_MARGIN:
                    best = int(j)

        return int(xs[best]), int(ys[best]), float(heat_norm[best]), float(center_norm[best]), True, relaxed

    def run(self, ctx):
        if ctx.tracks is None or len(ctx.tracks) == 0:
            ctx.candidates = []
            ctx.best_candidate = None
            ctx.debug["num_tracks"] = 0
            ctx.debug["num_candidates"] = 0
            ctx.debug["skipped_no_skeleton"] = 0
            ctx.debug["skipped_boundary"] = 0
            self._prev_id = None
            self._prev_xy = {}
            logger.debug(f"No tracks available. Skeleton: {ctx.skeleton is not None}, Skeleton empty: {ctx.skeleton is not None and np.sum(ctx.skeleton) == 0}")
            return ctx

        frame_h, frame_w = ctx.frame.shape[:2]
        frame_area = frame_h * frame_w
        max_dist = np.hypot(frame_w, frame_h) / 2
        heat_max_px = ctx.debug.get("heatmap_max_px", 0.0)

        all_candidates = []
        skipped_skeleton = 0  # tracks that had no skeleton pixel, fell back to the bbox centre
        skipped_boundary = 0  # tracks where the endpoint margin had to be relaxed

        ctx.debug["num_tracks"] = len(ctx.tracks)

        for xyxy, mask_i, confidence, class_id, tracker_id, data in ctx.tracks:
            if tracker_id is None:
                continue

            x1, y1, x2, y2 = map(float, xyxy)

            # aspect ratio
            width = x2 - x1
            height = y2 - y1

            if width <= 0 or height <= 0:
                continue

            aspect_ratio = width / height

            if not (ASPECT_RATIO_MIN <= aspect_ratio <= ASPECT_RATIO_MAX):
                continue  # Reject if aspect ratio not in range

            # size constraint
            bbox_area = width * height
            size_ratio = bbox_area / frame_area

            if not (BBOX_SIZE_MIN <= size_ratio <= BBOX_SIZE_MAX):
                continue  # Reject if size not in range

            norm_confidence = float(confidence) if confidence is not None else 0.0

            picked = self._pick_point(mask_i, (x1, y1, x2, y2), ctx, int(tracker_id), frame_w, frame_h)
            if picked is not None:
                px, py, norm_thickness, norm_center, on_skeleton, relaxed = picked
                skipped_boundary += int(relaxed)
            else:
                # No usable mask/skeleton: fall back to the bbox centre, with no thickness credit
                px, py = int((x1 + x2) / 2), int((y1 + y2) / 2)
                norm_thickness = 0.0
                norm_center = 1.0 - np.hypot(px - frame_w / 2, py - frame_h / 2) / max_dist
                on_skeleton = False
                skipped_skeleton += 1

            # weighted sum
            score = (
                WEIGHTS["confidence"] * norm_confidence
                + WEIGHTS["thickness"] * norm_thickness
                + WEIGHTS["center"] * norm_center
            )

            final_score = sigmoid(score * 2.0)

            heat_here = 0.0
            if on_skeleton and ctx.distance_heatmap is not None:
                heat_here = float(ctx.distance_heatmap[py, px]) * heat_max_px  # half-thickness in px

            candidate = {
                "track_id": int(tracker_id),
                "x": int(px),
                "y": int(py),
                "score": float(final_score),
                "confidence": float(norm_confidence),
                "center_distance": float(norm_center),
                "thickness": float(norm_thickness),
                "bbox_size": float(size_ratio),
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "distance_to_boundary": heat_here,
                "on_skeleton": on_skeleton,
            }

            all_candidates.append(candidate)

        # Best candidate overall (ties broken by lowest track id), sticky to last frame's track
        best = (
            max(all_candidates, key=lambda c: (c["score"], -c["track_id"]))
            if all_candidates
            else None
        )
        prev = next((c for c in all_candidates if c["track_id"] == self._prev_id), None)
        if best is not None and prev is not None and best is not prev \
                and best["score"] < prev["score"] + TRACK_SWITCH_MARGIN:
            best = prev

        self._prev_id = best["track_id"] if best is not None else None
        self._prev_xy = {c["track_id"]: (c["x"], c["y"]) for c in all_candidates}

        ctx.best_candidate = best
        ctx.candidates = all_candidates

        # Set debug counters
        ctx.debug["num_candidates"] = len(all_candidates)
        ctx.debug["num_confirmed_tracks"] = len(all_candidates)
        ctx.debug["skipped_no_skeleton"] = skipped_skeleton
        ctx.debug["skipped_boundary"] = skipped_boundary

        return ctx
