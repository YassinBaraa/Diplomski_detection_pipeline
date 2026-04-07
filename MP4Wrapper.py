import cv2
from FrameSource import FrameSource

class MP4Wrapper(FrameSource):

    def __init__(self, video_path):
        self.video_path = video_path
        cap = cv2.VideoCapture(video_path)

    def read(self):
        ret, frame = self.cap.read()

        return ret, frame
    
    def release(self):
        self.cap.release()

