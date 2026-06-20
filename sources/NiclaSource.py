import cv2
from sources.FrameSource import FrameSource
import serial
import time
import numpy as np

_BOOT_TIMEOUT = 8   # seconds to wait for first IMG: packet before attempting reset
_MAX_RESETS   = 2   # hard resets to attempt before giving up


class NiclaSource(FrameSource):
    def __init__(self, port='/dev/ttyACM0'):
        self._port = port
        self.ser = self._open_and_ensure_running()
        self._running = True
        self._recording = False
        self._video_writer = None
        self._output_path = None
        self._fps = 10

    def _open_and_ensure_running(self):
        """Open serial and hard-reset the Nicla if main.py isn't streaming."""
        ser = serial.Serial(self._port, 115200, timeout=1)
        time.sleep(2)

        # Probe: try to receive an IMG: header within _BOOT_TIMEOUT seconds
        for attempt in range(_MAX_RESETS + 1):
            deadline = time.time() + _BOOT_TIMEOUT
            while time.time() < deadline:
                line = ser.readline().decode(errors='ignore').strip()
                if line.startswith('IMG:'):
                    print(f"[NiclaSource] Streaming OK (attempt {attempt})")
                    # Put back the length token so read() can consume it normally
                    # by re-opening with the remaining buffer — simplest: just drain
                    # and rely on the next frame arriving immediately.
                    ser.reset_input_buffer()
                    return ser

            if attempt < _MAX_RESETS:
                print(f"[NiclaSource] No data — sending hard_reset (attempt {attempt + 1}/{_MAX_RESETS})")
                # Send Ctrl-C to stop any stuck REPL, then hard reset
                ser.write(b'\x03')
                time.sleep(0.3)
                ser.write(b'import pyb; pyb.hard_reset()\r\n')
                ser.flush()
                ser.close()
                time.sleep(4)  # wait for USB re-enumeration
                ser = serial.Serial(self._port, 115200, timeout=1)
                time.sleep(1)

        raise RuntimeError(
            "[NiclaSource] Nicla did not start streaming after reset. "
            "Check camera ribbon cable and VL53L1X sensor connection."
        )

    def read(self):
        while self._running:
            line = self.ser.readline().decode(errors='ignore').strip()
            if not line:
                continue
            if line.startswith("IMG:"):
                try:
                    length = int(line[4:])
                    jpg_data = self.ser.read(length)
                    dist_line = self.ser.readline().decode(errors='ignore').strip()
                    dist_mm = int(dist_line.split(":")[1])
                    img = cv2.imdecode(np.frombuffer(jpg_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if img is None:
                        continue
                    img = cv2.rotate(img, cv2.ROTATE_180)
                    if img is None:
                        continue

                    if self._recording:
                        if self._video_writer is None:
                            h, w = img.shape[:2]
                            fourcc = cv2.VideoWriter_fourcc(*'XVID')
                            self._video_writer = cv2.VideoWriter(
                                self._output_path, fourcc, self._fps, (w, h)
                            )
                        self._video_writer.write(img)

                    return True, img, {"distance_mm": dist_mm}
                except Exception as e:
                    print(f"[NiclaSource] Frame parse error: {e}")
                    continue

        return False, None, None  

    def release(self):
        self._running = False
        self.ser.close()

    def start_recording(self, output_path="recording.avi", fps=10):
        self._recording = True
        self._video_writer = None
        self._output_path = output_path
        self._fps = fps

    def stop_recording(self):
        self._recording = False
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None
            print(f"[NiclaSource] Recording saved to {self._output_path}")


"""
if __name__ == "__main__":
    source = NiclaSource()
    while True:
        ret, frame, meta = source.read()
        if ret:
            print(f"Distance: {meta['distance_mm']} mm")
            cv2.imshow("Nicla Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    source.release()
    cv2.destroyAllWindows()
"""