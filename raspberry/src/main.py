from data_reader import DataReader
from motors_control import MotorsControl
import sys
from local_inferencerer import LocalInferencer
import numpy as np
# import cv2


def main():
    # Creater reader objects
    print("START MAIN")
    yolo_reader = DataReader(port=8000, label="pixel")
    # print("YOLO WEBSOCKET READY")
    # # depth_reader = DataReader(port=8001, label="depth")
    # # speeds_reader = DataReader(port=8002, label="speeds")
    # print("DEPTH WEBSOCKET READY")
    motors_control = MotorsControl()
    # print("MAIN READY")
    motors_speed = motors_control.move_and_read([0, 0])
    inferencer = LocalInferencer()
    print("waiting for client")
    # inferencer.wait_to_client()
    print("Client detected")
    init_messages = 0
    try:
        while True:
            # human_pixel = None
            # Read yolo data
            # while human_pixel is None:
            human_pixel = yolo_reader.read_data()
            print(human_pixel)
            # if human_pixel is not None:
            #     print(human_pixel)
            # # Read depth data
            # depth_image = depth_reader.read_data()
            # obs = {
            #     "human_pixel": np.array(
            #         [1, 2],
            #         dtype=np.float64,
            #     ),
            #     "depth_image": np.array(
            #         [
            #             1,
            #             0,
            #             1,
            #             0,
            #             1,
            #             1,
            #             0,
            #             0,
            #         ],
            #         dtype=np.float64,
            #     ),
            #     "wheels_speed": np.array(
            #         motors_speed,
            #         dtype=np.float64,
            #     ),
            # }
            # while init_messages < 8:
            #     init_messages += 1
            #     inferencer.send_message(obs)

            # print(f"sending {obs}")
            # inferencer.send_message(obs)
            # print("sent")
            # action = inferencer.read_message()
            # print("action:", action)
            # print(depth_image)
            # cv2.imshow(depth_image)
            # cv2.waitKey(1)
            # Read motors speed
            # controller_speeds = speeds_reader.read_data()
            # Move motors
            # motors_speed = motors_control.move_and_read(action)
            # print("motors speed:", motors_speed)
            # Process to get motors speed

    except KeyboardInterrupt:
        print("Ctrl-C pressed!")
        sys.exit(0)


if __name__ == "__main__":
    print("prev func main")
    main()
