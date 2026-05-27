import numpy as np
import cv2
from hailo_platform import VDevice
from detectors.Detector import Detector

HEF_PATH    = "model/yolov8_segmentation.hef"
CONF_THRESH = 0.6
INPUT_SIZE  = 640


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))


def _dfl_decode(box_raw):
    H, W, _ = box_raw.shape
    box      = box_raw.reshape(H, W, 4, 16).astype(np.float32)
    bins     = np.arange(16, dtype=np.float32)
    box_exp  = np.exp(box - box.max(axis=-1, keepdims=True))
    box_soft = box_exp / box_exp.sum(axis=-1, keepdims=True)
    return (box_soft * bins).sum(axis=-1)   # [H, W, 4]


class HailoSegDetector(Detector):
    def __init__(self, hef_path=HEF_PATH, conf=CONF_THRESH):
        self.conf     = conf
        self.hef_path = hef_path

        print(f"[HailoSegDetector] Loading model: {hef_path}")
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
        returns: None, bboxes [N,4], masks [N,H,W] binary, confidences [N], class_ids [N]
        Masks are already resized to frame resolution.
        """
        orig_h, orig_w = frame.shape[:2]
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self._w, self._h))

        self._bindings.input().set_buffer(resized)
        self._configured.run([self._bindings], timeout=1000)

        float_out    = {k: v.astype(np.float32) for k, v in self._out_bufs.items()}
        dets, proto  = self._parse_seg_outputs(float_out, orig_w, orig_h)

        if not dets:
            empty = np.array([])
            empty_ids = np.array([], dtype=np.int32)
            return None, empty, None, empty, empty_ids

        bboxes      = np.array([[d[0], d[1], d[2], d[3]] for d in dets], dtype=np.float32)
        confidences = np.array([d[4] for d in dets],                      dtype=np.float32)
        class_ids   = np.zeros(len(dets), dtype=np.int32)

        # Build per-detection binary masks at frame resolution
        masks = None
        if proto is not None:
            masks = []
            for det in dets:
                coeffs = det[5]
                if coeffs is not None:
                    m      = _sigmoid(np.einsum('c,chw->hw', coeffs, proto))
                    m      = cv2.resize(m, (orig_w, orig_h))
                    masks.append((m > 0.5).astype(np.uint8))
                else:
                    masks.append(np.zeros((orig_h, orig_w), dtype=np.uint8))
            masks = np.array(masks)   # [N, orig_h, orig_w]

        return None, bboxes, masks, confidences, class_ids

    # ── internal helpers ────────────────────────────────────────────────────

    def _parse_seg_outputs(self, raw_outputs, orig_w, orig_h):
        proto     = None
        box_map   = {}
        cls_map   = {}
        coeff_map = {}

        for name, arr in raw_outputs.items():
            if arr.ndim == 4 and arr.shape[0] == 1:
                arr = arr.squeeze(0)
            if arr.ndim != 3:
                continue
            H, W, C = arr.shape

            if H == 160 and W == 160 and C == 32:
                proto = arr.transpose(2, 0, 1)          # [32, 160, 160]
            elif C == 64:
                box_map[INPUT_SIZE // H] = arr
            elif C == 1:
                cls_map[INPUT_SIZE // H] = arr[..., 0]
            elif C == 32:
                coeff_map[INPUT_SIZE // H] = arr

        if proto is None:
            print("[HailoSegDetector] Warning: proto tensor not found")
            return [], None

        detections = []
        for stride in [8, 16, 32]:
            if stride not in box_map or stride not in cls_map:
                continue
            box_raw = box_map[stride]
            scores  = _sigmoid(cls_map[stride])
            coeffs  = coeff_map.get(stride)
            H, W    = scores.shape
            ltrb    = _dfl_decode(box_raw)

            gy, gx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
            scale   = orig_w / INPUT_SIZE   # assumes square-ish input

            x1 = (gx - ltrb[..., 0]) * stride * (orig_w / INPUT_SIZE)
            y1 = (gy - ltrb[..., 1]) * stride * (orig_h / INPUT_SIZE)
            x2 = (gx + ltrb[..., 2]) * stride * (orig_w / INPUT_SIZE)
            y2 = (gy + ltrb[..., 3]) * stride * (orig_h / INPUT_SIZE)

            for iy, ix in np.argwhere(scores > self.conf):
                detections.append((
                    int(np.clip(x1[iy, ix], 0, orig_w)),
                    int(np.clip(y1[iy, ix], 0, orig_h)),
                    int(np.clip(x2[iy, ix], 0, orig_w)),
                    int(np.clip(y2[iy, ix], 0, orig_h)),
                    float(scores[iy, ix]),
                    coeffs[iy, ix].copy() if coeffs is not None else None
                ))

        if not detections:
            return [], proto

        boxes_np  = np.array([[d[0], d[1], d[2], d[3]] for d in detections], dtype=np.float32)
        scores_np = np.array([d[4] for d in detections],                      dtype=np.float32)
        indices   = cv2.dnn.NMSBoxes(boxes_np.tolist(), scores_np.tolist(), self.conf, 0.45)

        if len(indices) == 0:
            return [], proto

        return [detections[i] for i in indices.flatten()], proto

    def release(self):
        try:
            if self._configured is not None:
                self._configured.deactivate()
        except Exception as e:
            print(f"Error deactivating Hailo config: {e}")
        
        try:
            if self._device is not None:
                self._device.__exit__(None, None, None)
        except Exception as e:
            print(f"Error closing Hailo device: {e}")
