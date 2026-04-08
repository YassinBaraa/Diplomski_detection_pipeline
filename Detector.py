from ultralytics import YOLO
import cv2

MODEL_PATH = "model/best.pt"

class Detector:

    def __init__(self, model_path=MODEL_PATH):
        self.model = YOLO(model_path)

    def predict(self, frame):
        #returns segmentation masks, bounding boxes, and confidence scores for detected objects in the input frame
        results = self.model(frame)
        return results[0].boxes.data.cpu().numpy()