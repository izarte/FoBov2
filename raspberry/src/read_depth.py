import sys
import numpy as np
from websockets.sync.client import connect

import sys
import numpy as np
import ArducamDepthCamera as ac
# import cv2


MAX_DISTANCE = 4


class ReadDepth:
    def __init__(self):
        self.cam = ac.ArducamCamera()
        if self.cam.open(ac.TOFConnect.CSI, 0) != 0:
            print("initialization failed")
        if self.cam.start(ac.TOFOutput.DEPTH) != 0:
            print("Failed to start camera")
        self.data = []
        self.read_depth()

    def __del__(self):
        self.cam.stop()
        self.cam.close()

    def send_data(self):
        with connect("ws://main:8001/") as ws:
            ws.send(self.data)
            ws.close()

    def read_depth(self):
        import time

        while True:
            time.sleep(0.2)
            frame = self.cam.requestFrame(200)
            if frame is None:
                return
            depth_buf = frame.getDepthData()
            self.cam.releaseFrame(frame)
            depth_buf = (1 - (depth_buf / MAX_DISTANCE)) * 255
            depth_buf = np.clip(depth_buf, 0, 255).astype(np.uint8)
            # cv2.imshow("depth", depth_buf)
            # cv2.waitKey(1)
            image_bytes = depth_buf.tobytes()
            self.data = image_bytes
            self.send_data()


def main():
    try:
        detect = ReadDepth()
        # detect.detect_person()
    except KeyboardInterrupt:
        # detect.ws.close()
        sys.exit()


if __name__ == "__main__":
    main()
