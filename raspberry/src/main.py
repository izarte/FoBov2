from data_reader import DataReader
from motors_control import MotorsControl
import sys
from local_inferencerer import LocalInferencer
import numpy as np
import signal
# import cv2

MAX_SPEED = 20


def signal_handler(sig, frame):
    import sys

    print("Signal received:", sig)
    sys.exit(0)


def main():
    # Creater reader objects
    print("START MAIN")
    yolo_reader = DataReader(port=8000, label="pixel")
    # print("YOLO WEBSOCKET READY")
    depth_reader = DataReader(port=8001, label="depth")
    # # # speeds_reader = DataReader(port=8002, label="speeds")
    # # print("DEPTH WEBSOCKET READY")
    motors_control = MotorsControl()
    # # print("MAIN READY")
    motors_speed = motors_control.move_and_read([0, 0])
    signal.signal(signal.SIGTERM, signal_handler)
    inferencer = LocalInferencer()
    print("waiting for client")
    inferencer.wait_to_client()
    print("Client detected")
    init_messages = 0
    human_pixel = {"x": 0, "y": 0}
    try:
        while True:
            # human_pixel = None
            # Read yolo data
            # human_pixel = yolo_reader.read_data()
            # print("pixel: ", human_pixel)
            # Read depth data
            human_pixel = yolo_reader.read_data()
            print("pixel: ", human_pixel)
            # Read depth data
            depth_image = depth_reader.read_data()
            print("read detph")
            obs = {
                "human_pixel": np.array(
                    [human_pixel["x"], human_pixel["y"]],
                    dtype=np.float32,
                ),
                "depth_image": depth_image,
                "wheels_speed": np.array(
                    motors_speed,
                    dtype=np.float32,
                ),
            }
            while init_messages < 4:
                init_messages += 1
                inferencer.send_message(obs)
            inferencer.send_message(obs)
            print("sent")
            action = inferencer.read_message()
            print("action:", action)
            # print(depth_image)
            # cv2.imshow(depth_image)
            # cv2.waitKey(1)
            # Read motors speed
            # controller_speeds = speeds_reader.read_data()
            # Scale action
            action = np.array(action) * MAX_SPEED
            # Move motors
            motors_speed = motors_control.move_and_read(action)
            # print("motors speed:", motors_speed)

    except KeyboardInterrupt:
        print("Ctrl-C pressed!")
        sys.exit(0)


if __name__ == "__main__":
    print("prev func main")
    main()
