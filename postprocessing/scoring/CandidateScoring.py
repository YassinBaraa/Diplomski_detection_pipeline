import numpy as np
import logging
from pipeline.Stage import Stage

logger = logging.getLogger(__name__)


# Constraints (hard filters) - disabled to allow all tracks through
ASPECT_RATIO_MIN = 0.0  # Allow any aspect ratio
ASPECT_RATIO_MAX = 10.0
BBOX_SIZE_MIN = 0.0     # Allow any size
BBOX_SIZE_MAX = 1.0

# Weights for scoring variables (easy to adjust)
WEIGHTS = {
    "confidence": 0.4,       # Higher confidence is better
    "center_distance": 0.35, # Closer to center is better
    "bbox_size": 0.25,       # Size in target range is better
}

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))


class CandidateScoring(Stage):

    def run(self, ctx):
        if ctx.tracks is None or len(ctx.tracks) == 0:
            ctx.candidates = []
            ctx.best_candidate = None
            ctx.debug["num_tracks"] = 0
            ctx.debug["num_candidates"] = 0
            ctx.debug["skipped_no_skeleton"] = 0
            ctx.debug["skipped_boundary"] = 0
            logger.debug(f"No tracks available. Skeleton: {ctx.skeleton is not None}, Skeleton empty: {ctx.skeleton is not None and np.sum(ctx.skeleton) == 0}")
            return ctx

        frame_h, frame_w = ctx.frame.shape[:2]
        frame_area = frame_h * frame_w
        frame_diag = np.sqrt(frame_w**2 + frame_h**2)

        all_candidates = []
        skipped_skeleton = 0
        skipped_boundary = 0
        
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

            # normalize variables

            # Variable 1: Confidence (already in [0, 1])
            norm_confidence = float(confidence) if confidence is not None else 0.0

            # Variable 2: Distance to frame center
            bbox_center_x = (x1 + x2) / 2
            bbox_center_y = (y1 + y2) / 2
            frame_center_x = frame_w / 2
            frame_center_y = frame_h / 2

            dist_to_center = np.sqrt(
                (bbox_center_x - frame_center_x) ** 2
                + (bbox_center_y - frame_center_y) ** 2
            )
            max_dist = frame_diag / 2
            norm_center_distance = 1.0 - (dist_to_center / max_dist)

            # Variable 3: Size in target range (15-40%)
            # Score is highest in middle of range, lower at edges
            mid_size = (BBOX_SIZE_MIN + BBOX_SIZE_MAX) / 2
            norm_bbox_size = 1.0 - abs(size_ratio - mid_size) / (
                mid_size - BBOX_SIZE_MIN
            )
            norm_bbox_size = np.clip(norm_bbox_size, 0, 1)

            # weighted sum
            score = (
                WEIGHTS["confidence"] * norm_confidence
                + WEIGHTS["center_distance"] * norm_center_distance
                + WEIGHTS["bbox_size"] * norm_bbox_size
            )

            final_score = sigmoid(score * 2.0)

            # store candidate
            candidate = {
                "track_id": int(tracker_id),
                "x": int(bbox_center_x),
                "y": int(bbox_center_y),
                "score": float(final_score),
                "confidence": float(norm_confidence),
                "center_distance": float(norm_center_distance),
                "bbox_size": float(norm_bbox_size),
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "distance_to_boundary": float(dist_to_center),
            }

            all_candidates.append(candidate)

        # Find best candidate overall
        ctx.best_candidate = (
            max(all_candidates, key=lambda c: c["score"])
            if all_candidates
            else None
        )
        ctx.candidates = all_candidates

        # Set debug counters
        ctx.debug["num_candidates"] = len(all_candidates)
        ctx.debug["num_confirmed_tracks"] = len(all_candidates)
        ctx.debug["skipped_no_skeleton"] = skipped_skeleton
        ctx.debug["skipped_boundary"] = skipped_boundary

        return ctx
