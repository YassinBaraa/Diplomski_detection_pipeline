import cv2
from FrameSource import FrameSource


class DetectionPipeline:
    def __init__(self, source, detector, postprocessor):
        self.source = source
        self.detector = detector
        self.postprocessor = postprocessor

    def run(self):
        while True:
            ret, frame = self.source.read()
            if not ret:
                break

            detections = self.detector.predict(frame)
            result = self.postprocessor.process(detections, frame=frame)

            return result
            #yield result #return values one at a time instead of all at once
        



