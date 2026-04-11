import cv2
import numpy as np
from picamera2 import Picamera2
from FrameSource import FrameSource

class CameraSource(FrameSource):

    def __init__(self):
        picam2 = Picamera2()
        picam2.preview_configuration.main.size = (1280, 720)
        picam2.preview_configuration.main.format = "RGB888"
        picam2.preview_configuration.align()
        picam2.configure("preview")
        picam2.start()

    def read(self):
        frame = self.picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  
        return True, frame

    def release(self):
        self.picam2.stop()
        self.picam2.close()  # Extra safety