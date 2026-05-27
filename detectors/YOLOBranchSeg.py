from ultralytics import YOLO
import numpy as np

from detectors.Detector import Detector


MODEL_PATH = "model/best_small.pt"

class YOLOBranchSeg(Detector):
    def __init__(self, model_path=MODEL_PATH, conf=0.7):
        self.model = YOLO(model_path)
        self.conf = conf

    def predict(self, frame):
        results = self.model(frame, conf=self.conf)

        bboxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else np.array([])
        masks = results[0].masks.data.cpu().numpy() if results[0].masks is not None else None
        confidences = results[0].boxes.conf.cpu().numpy() if results[0].boxes is not None else np.array([])
        class_ids = results[0].boxes.cls.cpu().numpy().astype(np.int32) if results[0].boxes is not None else np.array([], dtype=np.int32)

        return results[0], bboxes, masks, confidences, class_ids
