import cv2
from sources.FrameSource import FrameSource
import serial
import time
import numpy as np

class NiclaSource(FrameSource):
    def __init__(self):
        time.sleep(3)  # let Nicla boot
        self.ser = serial.Serial('/dev/ttyACM0', 115200, timeout=2)

    def read(self):
        line = self.ser.readline().decode(errors='ignore').strip()
        if line.startswith("IMG:"):
            length = int(line[4:])
            jpg_data = self.ser.read(length)
            dist_line = self.ser.readline().decode(errors='ignore').strip()
            dist_mm = int(dist_line.split(":")[1])

            img = cv2.imdecode(np.frombuffer(jpg_data, dtype=np.uint8), cv2.IMREAD_COLOR)

            return True, img, {"distance_mm": dist_mm}
        
        return False, None, None

    def release(self):
        self.ser.close()

    def start_recording(self):
        # No recording functionality for NiclaSource
        pass

    def stop_recording(self):
        # No recording functionality for NiclaSource
        pass


"""
if __name__ == "__main__":
    source = NiclaSource(None)
    while True:
        ret, frame, dist_mm = source.read()
        if ret:
            print(f"Distance: {dist_mm} mm")
            cv2.imshow("Nicla Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    source.release()
    cv2.destroyAllWindows()
"""