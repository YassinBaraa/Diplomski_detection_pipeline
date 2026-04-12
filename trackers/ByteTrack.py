import supervision as sv
import cv2

from trackers.Tracker import Tracker


class ByteTrack(Tracker):
    def __init__(self, detector, tracker_model=None, label_annotator=None):
        super().__init__()
        self.tracker = tracker_model if tracker_model is not None else sv.ByteTrack()
        self.label_annotator = label_annotator if label_annotator is not None else sv.LabelAnnotator()
        self.detector = detector

    def track(self, frame, results):

        track = sv.Detections.from_ultralytics(results)
        track = self.tracker.update_with_detections(track)

        # Convert supervision Detections to iterable format for downstream processing
        # Format: (xyxy, mask, confidence, class_id, tracker_id, data)
        tracks_list = []
        for i in range(len(track)):
            xyxy = track.xyxy[i]
            mask = track.mask[i] if track.mask is not None else None
            confidence = track.confidence[i] if track.confidence is not None else 0.0
            class_id = track.class_id[i] if track.class_id is not None else 0
            tracker_id = track.tracker_id[i] if track.tracker_id is not None else None
            data = {}

            tracks_list.append((xyxy, mask, confidence, class_id, tracker_id, data))

        return tracks_list