import cv2
from FrameSource import FrameSource


class DetectionPipeline:
    def __init__(self, source: FrameSource, Detector, KeypointSelector):
        self.source = source
        self.detector = Detector
        self.keypoint_selector = KeypointSelector
    
    def run(self):
        while True:
            ret, frame = self.source.read()
            if not ret:
                break

            detections = self.detector.predict(frame)
            point = self.keypoint_selector.select_point(detections)

