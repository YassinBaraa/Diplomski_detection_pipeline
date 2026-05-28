from ultralytics import YOLO
import numpy as np
import cv2

from detectors.Detector import Detector


MODEL_PATH = "model/best_small.pt"

class YOLOBranchSeg(Detector):
    def __init__(self, model_path=MODEL_PATH, conf=0.7):
        self.model = YOLO(model_path)
        self.conf = conf

    def predict(self, frame):
        orig_h, orig_w = frame.shape[:2]
        results = self.model(frame, conf=self.conf)

        bboxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else np.array([])
        masks = results[0].masks.data.cpu().numpy() if results[0].masks is not None else None
        confidences = results[0].boxes.conf.cpu().numpy() if results[0].boxes is not None else np.array([])
        class_ids = results[0].boxes.cls.cpu().numpy().astype(np.int32) if results[0].boxes is not None else np.array([], dtype=np.int32)

        # Resize masks to match frame dimensions if needed
        if masks is not None and len(masks) > 0:
            mask_h, mask_w = masks[0].shape[:2]
            if mask_h != orig_h or mask_w != orig_w:
                resized_masks = []
                for mask in masks:
                    resized_mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                    resized_masks.append(resized_mask)
                masks = np.array(resized_masks)

        return results[0], bboxes, masks, confidences, class_ids
