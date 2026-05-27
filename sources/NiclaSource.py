import cv2
from sources.FrameSource import FrameSource
import serial
import time
import numpy as np


class NiclaSource(FrameSource):
    def __init__(self):
        time.sleep(3)  # let Nicla boot
        self.ser = serial.Serial('/dev/ttyACM0', 115200, timeout=2)
        self._running = True
        self._recording = False
        self._video_writer = None
        self._output_path = None
        self._fps = 10

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