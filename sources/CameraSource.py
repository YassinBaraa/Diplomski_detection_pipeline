import cv2
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FileOutput
from sources.FrameSource import FrameSource
import os

class CameraSource(FrameSource):
    def __init__(self):
        self.picam2 = Picamera2()
        self.picam2.preview_configuration.main.size = (1280, 720)
        self.picam2.preview_configuration.main.format = "RGB888"
        self.picam2.preview_configuration.align()
        self.picam2.configure("preview")
        self.picam2.start()
        self.encoder = None
        self.output = None

    def read(self):
        frame = self.picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return True, frame, None

    def release(self):
        self.stop_recording()
        self.picam2.stop()
        self.picam2.close()

    def start_recording(self, output_path="/home/baraa/Desktop/Diplomski/Diplomski_rad/detection_pipeline/sources"):
        try:
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            filepath = os.path.join(output_path, "recording.h264")
            self.encoder = H264Encoder()
            self.output = FileOutput(filepath)
            self.picam2.start_recording(self.encoder, self.output)
        except Exception as e:
            print(f"Error starting recording: {e}")

    def stop_recording(self):
        if self.encoder is not None and self.output is not None:
            try:
                self.picam2.stop_recording()
            except Exception as e:
                print(f"Error stopping recording: {e}")
            self.encoder = None
            self.output = None
