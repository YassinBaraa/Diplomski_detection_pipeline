import cv2
import numpy as np
import supervision as sv

# Constant-velocity Kalman filter on the box center (state: cx, cy, vx, vy).
# The motion model is linear, so this is a plain KF (an EKF would be identical).
F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
Q = np.eye(4) * 1.0    # process noise
R = np.eye(2) * 4.0    # measurement noise
VEL_DAMPING = 0.7      # velocity shrinks each coasted frame, so predictions settle instead of drifting
MIN_HITS = 5           # a track must have been seen this many frames before it may coast


class _Track:
    def __init__(self, cx, cy):
        self.x = np.array([cx, cy, 0, 0], dtype=float)
        self.P = np.eye(4) * 10.0
        self.misses = 0
        self.hits = 0

    def predict(self):
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, cx, cy):
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (np.array([cx, cy]) - H @ self.x)
        self.P = (np.eye(4) - K @ H) @ self.P


class TrackCoaster:
    """Wraps a tracker. When a track disappears, keeps emitting it at the
    Kalman-predicted position for up to `max_coast` frames.
    To remove: delete this file and the wrapper line in test_record.py."""

    def __init__(self, tracker, max_coast=5):
        self.tracker = tracker
        self.max_coast = max_coast
        self.state = {}  # tracker_id -> (_Track, last xyxy, last mask, conf, class_id)

    def track(self, frame, **kwargs):
        tracks, sv_det = self.tracker.track(frame, **kwargs)
        out = list(tracks)
        seen = set()

        for xyxy, mask, conf, class_id, tid, data in tracks:
            seen.add(tid)
            cx, cy = (xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2
            if tid not in self.state:
                self.state[tid] = [_Track(cx, cy), None, None, 0, 0]
            kf = self.state[tid][0]
            kf.predict()
            kf.update(cx, cy)
            kf.misses = 0
            kf.hits += 1
            self.state[tid][1:] = [xyxy, mask, conf, class_id]

        for tid in list(self.state):
            if tid in seen:
                continue
            kf, xyxy, mask, conf, class_id = self.state[tid]
            kf.misses += 1
            if kf.misses > self.max_coast or kf.hits < MIN_HITS:
                del self.state[tid]
                continue
            kf.x[2:] *= VEL_DAMPING
            kf.predict()
            # shift last box (and mask) to the predicted center
            dx = kf.x[0] - (xyxy[0] + xyxy[2]) / 2
            dy = kf.x[1] - (xyxy[1] + xyxy[3]) / 2
            new_xyxy = np.array(xyxy, dtype=np.float32) + np.array([dx, dy, dx, dy], dtype=np.float32)
            new_mask = None
            if mask is not None:
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                new_mask = cv2.warpAffine(mask.astype(np.uint8), M, mask.shape[::-1]) > 0
            out.append((new_xyxy, new_mask, conf, class_id, tid, {"coasted": True}))

        return out, self._to_detections(out, sv_det)

    @staticmethod
    def _to_detections(tracks, sv_det):
        if len(tracks) == 0:
            return sv_det
        has_mask = all(t[1] is not None for t in tracks)
        return sv.Detections(
            xyxy=np.array([t[0] for t in tracks], dtype=np.float32),
            mask=np.array([t[1] for t in tracks], dtype=bool) if has_mask else None,
            confidence=np.array([t[2] for t in tracks], dtype=np.float32),
            class_id=np.array([t[3] for t in tracks], dtype=np.int32),
            tracker_id=np.array([t[4] for t in tracks], dtype=int),
        )
