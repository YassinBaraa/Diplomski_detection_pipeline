import numpy as np
import cv2
import logging
from pipeline.Stage import Stage

logger = logging.getLogger(__name__)


# Constraints (hard filters) - disabled to allow all tracks through
ASPECT_RATIO_MIN = 0.0  # Allow any aspect ratio
ASPECT_RATIO_MAX = float("inf")  # was 10, which dropped long horizontal branches (w/h > 10)
BBOX_SIZE_MIN = 0.0     # Allow any size
BBOX_SIZE_MAX = 1.0

# Orientation filter: only branches running (roughly) horizontal or vertical in the image
# are candidates. The direction comes from a line fit through the track's skeleton pixels.
ORIENTATION_FILTER = True
ORIENTATION_TOLERANCE_DEG = 20.0  # accepted: angle <= this (horizontal) or >= 90 - this (vertical)
ORIENTATION_MIN_PIXELS = 10       # fewer skeleton pixels than this -> fit the track's mask instead
# Vertical branches only (the pilot yaws so the target branch hangs vertically in the image): a
# branch is a candidate only if it is within VERTICAL_TOLERANCE_DEG of vertical; one too small to
# tell its direction is left out too. Horizontal structures -- the ceiling beam, which the network
# calls a branch (0.59) and which nearly won on 2026-10-13 (0.834 vs the stick's 0.840) -- never
# become the target; their masks still count as obstacles for the crossing rule.
VERTICAL_ONLY = True
VERTICAL_TOLERANCE_DEG = 30.0

# Weights for the per-track score (easy to adjust)
WEIGHTS = {
    "confidence": 0.4,  # Higher detection confidence is better
    "thickness": 0.35,  # Chosen point sits on a thick part of the branch
    "center": 0.25,     # Chosen point is close to the image centre (less corrective motion)
}

# Which skeleton pixel of a track becomes its perch point
PIXEL_WEIGHTS = {"thickness": 0.6, "center": 0.4}
END_MARGIN_PX = 30         # keep the point this far from skeleton endpoints (gripper needs branch on both sides)
# ... and from the branch's own two ends (not where it leaves the image) at least this many widths:
# with only 30 px the 2026-10-13 10:19 flight locked on the very tip of a short stub, where the
# width can't be measured (size on 12 % of packets). No room for that -> no point on this branch.
END_MARGIN_WIDTHS = 5.0
FRAME_EDGE_END_PX = 3      # a skeleton end this close to the frame edge is where the branch leaves the image
TRACK_SWITCH_MARGIN = 0.05 # a rival track must beat the previous best by this much to take over
POINT_KEEP_RADIUS = 60     # px, previous point follows the branch to the nearest valid skeleton pixel within this
POINT_KEEP_MARGIN = 0.3    # ... as long as it scores at most this much below the new best pixel
                           # (tuned on the recordings: 6 px / 0.03 made the point jump 30+ px between frames)

# Keep the point away from crossings: other segmented structures (a pipe, a wire), junctions of
# the skeleton, and bulges of the mask (an unsegmented crossing makes it locally thicker). The
# merged mask is thickest there, so the thickness score picked exactly those spots (2026-10-11/12
# flights: the stick where it crosses the ceiling wire and pipe), where the gripper would hit them.
CROSSING_CLEARANCE = 10.0  # x the branch's median half-width (= 5 widths, room for the gripper), at least CROSSING_MIN_PX
CROSSING_MIN_PX = 15
BULGE_RATIO = 1.6          # half-width this much above the branch's median counts as a bulge
# A crossing the detector did not segment (the ceiling pipe on 2026-10-12): the background on BOTH
# sides of the branch is darker there than along the rest of it
SIDE_OFFSET = 6.0          # sample the background this many half-widths (+4 px) off the branch axis
SIDE_DARKER = 25           # gray levels below the branch's typical background, on both sides


NO_ROOM = "no_room"   # _pick_point: the branch has no point far enough from its ends


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))


# 3x3 sum = pixel + its 8-neighbours. The kernel must be float: with a uint8 kernel cv2.filter2D (4.10)
# returns wrong sums (1 along a line instead of 3), so no end points and no junctions were found.
_NEIGHBOURS = np.ones((3, 3), np.float32)


def _endpoints(sk):
    """Skeleton pixels with exactly one 8-neighbour. Returns [E, 2] (row, col)."""
    neighbours = cv2.filter2D(sk.astype(np.uint8), -1, _NEIGHBOURS, borderType=cv2.BORDER_CONSTANT)
    return np.argwhere(sk & (neighbours == 2))


def _orientation_ok(angle):
    """Whether a branch at `angle` (deg from horizontal, None = unknown) may be a candidate."""
    if VERTICAL_ONLY:
        return angle is not None and angle >= 90.0 - VERTICAL_TOLERANCE_DEG
    return angle is None or not ORIENTATION_TOLERANCE_DEG < angle < 90.0 - ORIENTATION_TOLERANCE_DEG


def _branch_angle(mask_i, skeleton, xyxy, frame_w, frame_h):
    """Angle of the branch axis from horizontal in degrees (0 = horizontal, 90 = vertical),
    or None if the track has too few pixels to tell."""
    if mask_i is None:
        return None
    x1, y1, x2, y2 = xyxy
    x1, y1 = max(int(x1), 0), max(int(y1), 0)
    x2, y2 = min(int(x2) + 1, frame_w), min(int(y2) + 1, frame_h)
    m = mask_i[y1:y2, x1:x2].astype(bool)
    pts = np.argwhere((skeleton[y1:y2, x1:x2] > 0) & m) if skeleton is not None else np.empty((0, 2))
    if len(pts) < ORIENTATION_MIN_PIXELS:
        pts = np.argwhere(m)
    if len(pts) < ORIENTATION_MIN_PIXELS:
        return None
    vx, vy = cv2.fitLine(pts[:, ::-1].astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)[:2]
    return float(np.degrees(np.arctan2(abs(vy), abs(vx))))


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

    @staticmethod
    def _crossed(ctx, pts, x1, y1, med_half):
        """(rows, cols) of the skeleton pixels `pts` (ROI coords) where something crosses the branch:
        the background on both sides, SIDE_OFFSET half-widths off the axis, is clearly darker than
        the branch's typical background."""
        if len(pts) < 10:
            return pts[:0, 0], pts[:0, 1]
        gray = ctx.debug.get("_gray")
        if gray is None:
            gray = ctx.debug["_gray"] = cv2.cvtColor(ctx.frame, cv2.COLOR_BGR2GRAY)
        vx, vy = cv2.fitLine(pts[:, ::-1].astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)[:2]
        n = np.array([-vy, vx])
        s = SIDE_OFFSET * med_half + 4.0
        h, w = gray.shape
        xy = pts[:, ::-1].astype(np.float32) + (x1, y1)
        sides = []
        for sign in (1.0, -1.0):
            q = np.round(xy + sign * s * n).astype(int)
            q[:, 0], q[:, 1] = np.clip(q[:, 0], 0, w - 1), np.clip(q[:, 1], 0, h - 1)
            sides.append(gray[q[:, 1], q[:, 0]].astype(np.float32))
        ref = float(np.median(np.concatenate(sides)))
        hit = (sides[0] < ref - SIDE_DARKER) & (sides[1] < ref - SIDE_DARKER)
        return pts[hit, 0], pts[hit, 1]

    def _pick_point(self, mask_i, xyxy, ctx, track_id, frame_w, frame_h, other_mask=None, tip_rule=True):
        """Returns (x, y, heat_norm, center_norm, on_skeleton, relaxed), None if no skeleton pixel, or
        NO_ROOM if every skeleton pixel is too close to the branch's ends. tip_rule=False drops the
        END_MARGIN_WIDTHS rule (END_MARGIN_PX from every end still applies): the point of a branch the
        operator chose although it has no room (live view).
        other_mask: pixels of the other segmented structures (bool, full frame), to stay away from."""
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

        # Stay away from the ends of the skeleton: END_MARGIN_PX from every end, and from the
        # branch's own two ends (the farthest-apart endpoints not at the frame edge) END_MARGIN_WIDTHS
        # widths beyond the skeleton end (which stops about half a width short of the tip).
        # Nothing left -> NO_ROOM: this branch gets no point (never the tip as a fallback).
        relaxed = False
        half_px = heat * ctx.debug.get("heatmap_max_px", 0.0)
        med_half_all = float(np.median(half_px)) if half_px.max() > 0 else 0.0
        ends = _endpoints(sk)
        keep = np.ones(len(pts), dtype=bool)
        if len(ends) > 0:
            dist = np.sqrt(((pts[:, None, :] - ends[None, :, :]) ** 2).sum(axis=2))   # [P, E]
            keep = dist.min(axis=1) >= END_MARGIN_PX
            if len(ends) >= 2:
                de = np.sqrt(((ends[:, None, :] - ends[None, :, :]) ** 2).sum(axis=2))
                tips = list(np.unravel_index(int(np.argmax(de)), de.shape))
            else:
                tips = [0]
            fy, fx = ends[:, 0] + y1, ends[:, 1] + x1
            at_frame_edge = (fx < FRAME_EDGE_END_PX) | (fy < FRAME_EDGE_END_PX) | \
                            (fx >= frame_w - FRAME_EDGE_END_PX) | (fy >= frame_h - FRAME_EDGE_END_PX)
            tip_margin = max(END_MARGIN_PX, END_MARGIN_WIDTHS * 2.0 * med_half_all) + med_half_all
            for e in (tips if tip_rule else []):
                if not at_frame_edge[e]:
                    keep &= dist[:, e] >= tip_margin
            if not keep.any():
                return NO_ROOM
        # Stay away from crossings, junctions and bulges, unless that would leave nothing
        med_half = float(np.median(half_px[keep])) if half_px.max() > 0 else 0.0
        if med_half > 0:
            blocked = np.zeros(sk.shape, dtype=bool)
            if other_mask is not None:
                blocked |= other_mask[y1:y2, x1:x2]
            sk_all = ctx.skeleton[y1:y2, x1:x2] > 0
            n_all = cv2.filter2D(sk_all.astype(np.uint8), -1, _NEIGHBOURS, borderType=cv2.BORDER_CONSTANT)
            blocked |= sk_all & (n_all >= 4)                       # pixel + 3 or more neighbours: junction
            bulge = pts[half_px > BULGE_RATIO * med_half]
            blocked[bulge[:, 0], bulge[:, 1]] = True
            blocked[self._crossed(ctx, pts, x1, y1, med_half)] = True
            if blocked.any():
                clear = cv2.distanceTransform((~blocked).astype(np.uint8), cv2.DIST_L2, 3)[pts[:, 0], pts[:, 1]]
                far = keep & (clear >= max(CROSSING_CLEARANCE * med_half, CROSSING_MIN_PX))
                if far.any():
                    keep = far
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
            ctx.debug["rejected_candidates"] = []
            self._prev_id = None
            self._prev_xy = {}
            logger.debug(f"No tracks available. Skeleton: {ctx.skeleton is not None}, Skeleton empty: {ctx.skeleton is not None and np.sum(ctx.skeleton) == 0}")
            return ctx

        frame_h, frame_w = ctx.frame.shape[:2]
        frame_area = frame_h * frame_w
        max_dist = np.hypot(frame_w, frame_h) / 2
        heat_max_px = ctx.debug.get("heatmap_max_px", 0.0)

        all_candidates = []
        # Tracks that are not candidates, with the reason (for the live view): {"track_id", "reason",
        # "bbox", "angle_deg"} -- "end" ones also carry a point without the tip rule, which only an
        # explicit choice of that branch uses.
        rejected = []
        skipped_skeleton = 0  # tracks that had no skeleton pixel, fell back to the bbox centre
        skipped_boundary = 0  # tracks left out: no point far enough from the branch's ends
        skipped_orientation = 0  # tracks rejected for being neither horizontal nor vertical

        ctx.debug["num_tracks"] = len(ctx.tracks)

        # How many segmented structures cover each pixel, so each track can see the others
        masks = [m for _, m, *_ in ctx.tracks if m is not None]
        cover = np.sum([m.astype(np.uint8) for m in masks], axis=0) if masks else None

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

            angle = _branch_angle(mask_i, ctx.skeleton, (x1, y1, x2, y2), frame_w, frame_h)
            if ORIENTATION_FILTER and not _orientation_ok(angle):
                skipped_orientation += 1
                rejected.append({"track_id": int(tracker_id), "reason": "direction", "angle_deg": angle,
                                 "bbox": (int(x1), int(y1), int(x2), int(y2)), "confidence": float(confidence or 0.0)})
                continue

            norm_confidence = float(confidence) if confidence is not None else 0.0

            other = (cover - mask_i.astype(np.uint8)) > 0 if cover is not None and mask_i is not None else None
            picked = self._pick_point(mask_i, (x1, y1, x2, y2), ctx, int(tracker_id), frame_w, frame_h, other)
            no_room = picked is NO_ROOM
            if no_room:
                skipped_boundary += 1   # no point clear of the branch's ends: not a candidate
                picked = self._pick_point(mask_i, (x1, y1, x2, y2), ctx, int(tracker_id), frame_w, frame_h, other,
                                          tip_rule=False)
                if picked is None or picked is NO_ROOM:
                    rejected.append({"track_id": int(tracker_id), "reason": "end", "angle_deg": angle,
                                     "bbox": (int(x1), int(y1), int(x2), int(y2)),
                                     "confidence": float(confidence or 0.0)})
                    continue
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
                "angle_deg": angle,
            }

            if no_room:
                candidate["reason"] = "end"
                rejected.append(candidate)
                continue
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
        ctx.debug["rejected_candidates"] = rejected

        # Set debug counters
        ctx.debug["num_candidates"] = len(all_candidates)
        ctx.debug["num_confirmed_tracks"] = len(all_candidates)
        ctx.debug["skipped_no_skeleton"] = skipped_skeleton
        ctx.debug["skipped_boundary"] = skipped_boundary
        ctx.debug["skipped_orientation"] = skipped_orientation

        return ctx
