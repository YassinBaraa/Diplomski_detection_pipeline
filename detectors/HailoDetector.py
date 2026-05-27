import numpy as np
import cv2
from hailo_platform import VDevice
from detectors.Detector import Detector

HEF_PATH    = "model/yolov8_detection.hef"
CONF_THRESH = 0.6
INPUT_SIZE  = 640


class HailoDetector(Detector):
    def __init__(self, hef_path=HEF_PATH, conf=CONF_THRESH):
        self.conf     = conf
        self.hef_path = hef_path

        print(f"[HailoDetector] Loading model: {hef_path}")
        self._device         = VDevice()
        self._infer_model    = self._device.create_infer_model(hef_path)
        self._h, self._w, _  = self._infer_model.input().shape

        self._configured = self._infer_model.configure().__enter__()
        self._configured.activate()

        self._out_bufs = {
            o.name: np.empty(o.shape, dtype=np.uint8)
            for o in self._infer_model.outputs
        }
        self._bindings = self._configured.create_bindings()
        for name, buf in self._out_bufs.items():
            self._bindings.output(name).set_buffer(buf)

    def predict(self, frame):
        """
        frame: BGR numpy array (any resolution)
        returns: bboxes [N,4], confidences [N], masks=None, class_ids [N]
        """
        orig_h, orig_w = frame.shape[:2]
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self._w, self._h))

        self._bindings.input().set_buffer(resized)
        self._configured.run([self._bindings], timeout=1000)

        first_out = list(self._out_bufs.values())[0]
        dets      = self._parse_nms_output(first_out, orig_w, orig_h)

        if not dets:
            empty = np.array([])
            return None, empty, None, empty, empty

        bboxes      = np.array([[d[0], d[1], d[2], d[3]] for d in dets], dtype=np.float32)
        confidences = np.array([d[4] for d in dets],                      dtype=np.float32)
        class_ids   = np.zeros(len(dets), dtype=np.float32)   # single-class model
        masks       = None

        return None, bboxes, masks, confidences, class_ids

    def _parse_nms_output(self, nms_output, orig_w, orig_h):
        detections = []
        flat = np.array(nms_output).flatten().astype(np.float32)
        if len(flat) == 0:
            return detections
        num_det = int(flat[0])
        for i in range(num_det):
            base = 1 + i * 5
            if base + 5 > len(flat):
                break
            y1, x1, y2, x2, score = flat[base:base + 5]
            if score < self.conf:
                continue
            detections.append((
                int(np.clip(x1 * orig_w, 0, orig_w)),
                int(np.clip(y1 * orig_h, 0, orig_h)),
                int(np.clip(x2 * orig_w, 0, orig_w)),
                int(np.clip(y2 * orig_h, 0, orig_h)),
                float(score)
            ))
        return detections

    def release(self):
        self._configured.deactivate()
        self._device.__exit__(None, None, None)