from ultralytics import YOLO
import numpy as np

from detectors.Detector import Detector

MODEL_PATH = "model/best_detection.pt"

class YoloDetector(Detector):
    
    def __init__(self, model_path=MODEL_PATH, conf=0.7):
        self.conf = conf
        self.model_path = model_path
        print(f"[YoloDetector] Loading model: {model_path}")
        self.model = YOLO(model_path)

    def predict(self, frame):

        results = self.model(frame, conf=self.conf)

        bboxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else np.array([])
        confidences = results[0].boxes.conf.cpu().numpy() if results[0].boxes is not None else np.array([])
        class_ids = results[0].boxes.cls.cpu().numpy() if results[0].boxes is not None else np.array([])
        masks = None  # No segmentation in detection mode

        return results[0], bboxes, masks, confidences, class_ids


