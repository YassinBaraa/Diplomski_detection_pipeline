import glob
import time
import cv2
from sources.FrameSource import FrameSource

# /dev/videoN numbering isn't guaranteed to survive a reboot/replug, but udev's
# by-id symlink is -- falls back to DEVICE_FALLBACK if no by-id match is found
# (e.g. a differently-branded UC60 clone that udev doesn't tag as DSJ).
DEVICE_BY_ID_GLOB = "/dev/v4l/by-id/*DSJ_UC60_Video*-video-index0"
DEVICE_FALLBACK = 0

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
# The full pipeline (detection + KLT tracking + recording) can't sustain the
# camera's native rate -- requesting 30fps let frames queue up in V4L2's
# buffer faster than they were drained, so target_point fell further and
# further behind real time as a run went on (traced live: lag grew from ~2s
# to over 20s within a single run). Capped to what the pipeline can actually
# keep up with, so there's less to queue in the first place.
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
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, 50)
        # Keep V4L2's internal queue at 1 frame: the pipeline (detection +
        # tracking + recording) can't sustain the camera's native rate, and
        # without this the queue backs up silently -- read() keeps returning
        # the OLDEST buffered frame, so the reported target_point falls
        # further and further behind real time the longer a run goes (traced
        # live: growing lag from ~2s to over 20s within a single run). This
        # drops stale frames instead of processing them late.
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[DSJSource] Opened {self.device!r} at {actual_w}x{actual_h}")

    def read(self):
        ret, frame = self.cap.read()
        # Grab time on this computer's monotonic clock, taken before any processing
        # and carried with the frame, so the UDP packet can report its "age".
        t_frame = time.monotonic()
        if not ret or frame is None:
            return False, None, None
        return True, frame, {"t_frame": t_frame}

    def release(self):
        self.cap.release()
