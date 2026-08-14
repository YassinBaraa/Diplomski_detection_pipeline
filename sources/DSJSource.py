import glob
import cv2
from sources.FrameSource import FrameSource

# /dev/videoN numbering isn't guaranteed to survive a reboot/replug, but udev's
# by-id symlink is -- falls back to DEVICE_FALLBACK if no by-id match is found
# (e.g. a differently-branded UC60 clone that udev doesn't tag as DSJ).
DEVICE_BY_ID_GLOB = "/dev/v4l/by-id/*DSJ_UC60_Video*-video-index0"
DEVICE_FALLBACK = 0

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS = 30


def _resolve_device():
    matches = sorted(glob.glob(DEVICE_BY_ID_GLOB))
    if matches:
        return matches[0]
    print(f"[DSJSource] No by-id match for {DEVICE_BY_ID_GLOB!r}, "
          f"falling back to device {DEVICE_FALLBACK!r}")
    return DEVICE_FALLBACK


class DSJSource(FrameSource):
    """USB UVC camera driver for the DSJ-3079-HE.

    Standard UVC (USB Video Class) device (Realtek UC60 bridge chip, USB ID
    0bda:5805) -- driven entirely by the kernel's uvcvideo driver via
    V4L2/OpenCV, no vendor SDK needed. It exposes two /dev/videoN nodes:
    index0 is the actual capture node (used here), index1 is metadata-only.

    Confirmed via `v4l2-ctl -d <dev> --list-formats-ext`: MJPG up to
    3264x2448@15fps, YUYV up to 1920x1080@5fps. MJPG is used here since it
    covers the full resolution range at higher framerates and OpenCV decodes
    it natively.
    """

    def __init__(self, device=None, width=FRAME_WIDTH, height=FRAME_HEIGHT, fps=FPS):
        self.device = device if device is not None else _resolve_device()
        self.cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"[DSJSource] Failed to open camera at {self.device!r}")

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[DSJSource] Opened {self.device!r} at {actual_w}x{actual_h}")

    def read(self):
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False, None, None
        return True, frame, None

    def release(self):
        self.cap.release()
