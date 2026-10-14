import numpy as np
import cv2
from hailo_platform import VDevice
from detectors.Detector import Detector

HEF_PATH    = "model/yolov8_segmentation.hef"
CONF_THRESH = 0.5
INPUT_SIZE  = 640
LETTERBOX_GREY = 114   # the grey YOLOv8 pads with in training


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

        # Keep the context manager: release() must __exit__ it before closing the device
        self._configured_cm  = self._infer_model.configure()
        self._configured     = self._configured_cm.__enter__()
        self._configured.activate()

        self._out_bufs = {
            o.name: np.empty(o.shape, dtype=np.uint8)
            for o in self._infer_model.outputs
        }
        self._quant = {
            o.name: (o.quant_infos[0].qp_scale, o.quant_infos[0].qp_zp)
            for o in self._infer_model.outputs
        }
        self._bindings = self._configured.create_bindings()
        for name, buf in self._out_bufs.items():
            self._bindings.output(name).set_buffer(buf)

    def predict(self, frame):
        orig_h, orig_w = frame.shape[:2]
        # Letterbox, as YOLOv8 is trained: scale evenly to fit (1280x720 -> 640x360) and pad with grey,
        # instead of squashing to 640x640 (squashed: lower confidence, fewer detections)
        scale   = min(self._w / orig_w, self._h / orig_h)
        new_w, new_h = int(round(orig_w * scale)), int(round(orig_h * scale))
        pad_x, pad_y = (self._w - new_w) // 2, (self._h - new_h) // 2
        small   = cv2.cvtColor(cv2.resize(frame, (new_w, new_h)), cv2.COLOR_BGR2RGB)
        inp     = cv2.copyMakeBorder(small, pad_y, self._h - new_h - pad_y, pad_x, self._w - new_w - pad_x,
                                     cv2.BORDER_CONSTANT, value=(LETTERBOX_GREY,) * 3)
        self._letterbox = (scale, pad_x, pad_y, new_w, new_h)

        self._bindings.input().set_buffer(inp)
        self._configured.run([self._bindings], timeout=1000)

        float_out   = {
            k: (v.astype(np.float32) - self._quant[k][1]) * self._quant[k][0]
            for k, v in self._out_bufs.items()
        }
        dets, proto = self._parse_seg_outputs(float_out, orig_w, orig_h)

        empty_boxes = np.array([], dtype=np.float32).reshape(0, 4)
        empty_conf  = np.array([], dtype=np.float32)
        empty_ids   = np.array([], dtype=np.int32)

        if not dets or proto is None:
            return None, empty_boxes, None, empty_conf, empty_ids

        bboxes      = np.array([[d[0], d[1], d[2], d[3]] for d in dets], dtype=np.float32)
        confidences = np.array([d[4] for d in dets], dtype=np.float32)
        class_ids   = np.zeros(len(dets), dtype=np.int32)

        # the photo's part of the prototype masks (160x160 over the 640x640 input, 4 input px per cell)
        _, pad_x, pad_y, new_w, new_h = self._letterbox
        cell  = self._w / proto.shape[2]
        proto = np.ascontiguousarray(proto[:, int(round(pad_y / cell)):int(round((pad_y + new_h) / cell)),
                                           int(round(pad_x / cell)):int(round((pad_x + new_w) / cell))])

        masks = []
        for det in dets:
            x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
            coeffs = det[5]

            if coeffs is None:
                masks.append(np.zeros((orig_h, orig_w), dtype=bool))
                continue

            m      = _sigmoid(np.einsum('c,chw->hw', coeffs, proto))  # [90,160] for 1280x720
            m      = cv2.resize(m, (orig_w, orig_h))                   # [orig_h, orig_w]
            binary = (m > 0.5)

            clipped = np.zeros((orig_h, orig_w), dtype=bool)
            clipped[y1:y2, x1:x2] = binary[y1:y2, x1:x2]
            masks.append(clipped)

        masks = np.array(masks, dtype=bool)  # [N, orig_h, orig_w]

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
            scores  = cls_map[stride]   # Hailo applies sigmoid inside HEF; values already in [0,1]
            coeffs  = coeff_map.get(stride)
            H, W    = scores.shape
            ltrb    = _dfl_decode(box_raw)

            # YOLOv8 measures the box edges from the cell's centre (cell + 0.5), not its corner
            gy, gx = np.meshgrid(np.arange(H) + 0.5, np.arange(W) + 0.5, indexing='ij')

            # network input pixels -> photo pixels (undo the letterbox)
            scale, pad_x, pad_y = self._letterbox[:3]
            x1 = ((gx - ltrb[..., 0]) * stride - pad_x) / scale
            y1 = ((gy - ltrb[..., 1]) * stride - pad_y) / scale
            x2 = ((gx + ltrb[..., 2]) * stride - pad_x) / scale
            y2 = ((gy + ltrb[..., 3]) * stride - pad_y) / scale

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

        # NMSBoxes takes (x, y, width, height)
        boxes_np  = np.array([[d[0], d[1], d[2] - d[0], d[3] - d[1]] for d in detections], dtype=np.float32)
        scores_np = np.array([d[4] for d in detections],                                      dtype=np.float32)
        indices   = cv2.dnn.NMSBoxes(boxes_np.tolist(), scores_np.tolist(), self.conf, 0.45)

        if len(indices) == 0:
            return [], proto

        return [detections[i] for i in indices.flatten()], proto

    def release(self):
        # Tear down innermost-first (bindings -> configured model -> infer model ->
        # device). Closing the device while the configured model was still alive
        # segfaulted on exit; never releasing it hung the process on exit.
        self._bindings = None
        try:
            if self._configured_cm is not None:
                self._configured_cm.__exit__(None, None, None)
        except Exception as e:
            print(f"Error closing Hailo configured model: {e}")
        self._configured_cm = None
        self._configured = None
        self._infer_model = None
        try:
            if self._device is not None:
                self._device.release()
        except Exception as e:
            print(f"Error closing Hailo device: {e}")
        self._device = None
