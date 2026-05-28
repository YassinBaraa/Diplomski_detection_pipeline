import numpy as np
from pipeline.Stage import Stage


# Constraints (hard filters)
ASPECT_RATIO_MIN = 1.2  # x = 1.3*y 
ASPECT_RATIO_MAX = 1.4
BBOX_SIZE_MIN = 0.15    # 15% of frame
BBOX_SIZE_MAX = 0.40    # 40% of frame

# Weights for scoring variables (easy to adjust)
WEIGHTS = {
    "confidence": 0.4,       # Higher confidence is better
    "center_distance": 0.35, # Closer to center is better
    "bbox_size": 0.25,       # Size in target range is better
}

# normalization
USE_SIGMOID = True


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))


def normalize_value(value, min_val, max_val):
    if min_val == max_val:
        return 0.5
    return np.clip((value - min_val) / (max_val - min_val), 0, 1)


class CandidateScoring(Stage):

    def run(self, ctx):
        if ctx.tracks is None or len(ctx.tracks) == 0:
            ctx.candidates = []
            ctx.best_candidate = None
            return ctx

        frame_h, frame_w = ctx.frame.shape[:2]
        frame_area = frame_h * frame_w
        frame_diag = np.sqrt(frame_w**2 + frame_h**2)

        all_candidates = []
        best_per_track = []

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

            # Apply sigmoid for smooth normalization
            if USE_SIGMOID:
                final_score = sigmoid(score * 2.0)  # Scale for sigmoid
            else:
                final_score = np.clip(score, 0, 1)

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
            }

            all_candidates.append(candidate)
            best_per_track.append(candidate)

        # Find best candidate overall
        ctx.best_candidate = (
            max(all_candidates, key=lambda c: c["score"])
            if all_candidates
            else None
        )
        ctx.candidates = best_per_track

        return ctx
