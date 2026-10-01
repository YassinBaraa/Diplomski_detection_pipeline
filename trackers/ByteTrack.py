import supervision as sv
import cv2
import numpy as np
from trackers.Tracker import Tracker


class ByteTrack(Tracker):
    def __init__(self, detector, tracker_model=None, label_annotator=None,
                 track_activation_threshold=0.25, lost_track_buffer=30,
                 minimum_consecutive_frames=1):
        super().__init__()
        # Detector confidence should be set low (e.g. 0.25): detections below
        # track_activation_threshold can only keep an existing track alive,
        # they can't start a new one (ByteTrack's two-stage association).
        self.tracker = tracker_model if tracker_model is not None else sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_consecutive_frames=minimum_consecutive_frames,
        )
        self.label_annotator = label_annotator if label_annotator is not None else sv.LabelAnnotator()
        self.box_annotator = sv.BoxAnnotator()
        self.mask_annotator = sv.MaskAnnotator()
        self.detector = detector

    def track(self, frame, results=None, bboxes=None, masks=None, confidences=None, class_ids=None):
        if isinstance(results, sv.Detections):
            detections = results
        elif bboxes is not None and len(bboxes) > 0:
            if confidences is None or len(confidences) == 0:
                confidences = np.zeros(len(bboxes), dtype=np.float32)
            if class_ids is None or len(class_ids) == 0:
                class_ids = np.zeros(len(bboxes), dtype=np.int32)
            if masks is not None and len(masks) == 0:
                masks = None

            # Cast masks to bool to satisfy supervision >= 0.28.0
            if masks is not None and masks.dtype != bool:
                masks = masks > 0.5

            detections = sv.Detections(
                xyxy=np.array(bboxes, dtype=np.float32),
                mask=masks,
                confidence=np.array(confidences, dtype=np.float32),
                class_id=np.array(class_ids, dtype=np.int32),
            )
        elif results is not None:
            detections = sv.Detections.from_ultralytics(results)
        else:
            return [], sv.Detections.empty()

        track = self.tracker.update_with_detections(detections)

        tracks_list = []
        for i in range(len(track)):
            xyxy        = track.xyxy[i]
            mask        = track.mask[i] if track.mask is not None else None
            confidence  = track.confidence[i] if track.confidence is not None else 0.0
            class_id    = track.class_id[i] if track.class_id is not None else 0
            tracker_id  = track.tracker_id[i] if track.tracker_id is not None else None
            data        = {}
            tracks_list.append((xyxy, mask, confidence, class_id, tracker_id, data))

        return tracks_list, track  # also return sv.Detections for annotation