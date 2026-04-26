import cv2
from picamera2 import Picamera2
from detection_pipeline_ros.sources.FrameSource import FrameSource

class CameraSource(FrameSource):
    def __init__(self, resolution_width=1280, resolution_height=720, format="RGB888"):
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(
            main={"size": (resolution_width, resolution_height), "format": format}
        )
        self.picam2.configure(config)
        self.picam2.start()

    def read(self):
        frame = self.picam2.capture_array()
        if frame is None:
            return False, None
        return True, frame

    def release(self):
        self.picam2.close()
