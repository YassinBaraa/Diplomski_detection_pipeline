import cv2
import numpy as np
from picamera2 import Picamera2
from FrameSource import FrameSource

class MP4Wrapper(FrameSource):

    def __init__(self):
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(main={"size": (640, 480)}) 
        self.picam2.configure(config)
        self.picam2.start()

    def read(self):
        request = self.picam2.capture_request()
        frame = request.make_array("main")
        request.release()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  
        return True, frame

    def release(self):
        self.picam2.stop()
        self.picam2.close()  # Extra safety