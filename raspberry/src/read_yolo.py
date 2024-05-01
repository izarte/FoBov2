#!/usr/bin/env python3.8
import sys
import cv2
from websocket import create_connection
import json
from ultralytics import YOLO


class DetectPerson:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Can't open camera")
            exit()
        self.model = YOLO("yolov8n_int8.tflite", task="track")
        # self.ws = create_connection("ws://localhost:8000/")
        self.data = {"x": 0, "y": 0}
        self.detect_person()

    def __del__(self):
        self.cap.release()

    def send_data(self):
        ws = create_connection("ws://localhost:8000/")
        ws.send(json.dumps(self.data))
        ws.close()

    def detect_person(self):
        # Actually detect blue center and calculates its difference
        # t = time.time()
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Can't see")

            result = self.model.track(
                source=frame,
                # task='detect',
                agnostic_nms=True,
                classes=0,
                # tracker="botsort.yaml",
                half=True,
                verbose=False,
            )[0]
            boxes = result.boxes.cpu()
            if result.boxes.id is not None:
                x = int(
                    boxes.xyxy[0][0]
                    + ((boxes.xyxy[0][2] - boxes.xyxy[0][0]).item() / 2)
                )
                y = int(
                    boxes.xyxy[0][1]
                    + ((boxes.xyxy[0][3] - boxes.xyxy[0][1]).item() / 2)
                )
                # print(x, y)
                self.data["x"] = x
                self.data["y"] = y
            else:
                self.data = {"x": -1, "y": -1}
            # print(self.data['x'], self.data['y'])

            # print("FPS: ", 1 / (time.time() - t))
            self.send_data()


def main():
    try:
        detect = DetectPerson()
        # detect.detect_person()
    except KeyboardInterrupt:
        # detect.ws.close()
        sys.exit()


if __name__ == "__main__":
    main()
