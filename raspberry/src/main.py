from data_reader import DataReader
from motors_control import MotorsControl
import sys

# import cv2


def main():
    # Creater reader objects
    print("START MAIN")
    yolo_reader = DataReader(port=8000, label="pixel")
    print("YOLO WEBSOCKET READY")
    depth_reader = DataReader(port=8001, label="depth")
    speeds_reader = DataReader(port=8002, label="speeds")
    print("DEPTH WEBSOCKET READY")
    motors_control = MotorsControl()
    print("MAIN READY")

    try:
        while True:
            # human_pixel = None
            # Read yolo data
            # while human_pixel is None:
            # human_pixel = yolo_reader.read_data()
            # if human_pixel is not None:
            #     print(human_pixel)
            # # Read depth data
            # depth_image = depth_reader.read_data()
            # print(depth_image)
            # cv2.imshow(depth_image)
            # cv2.waitKey(1)
            # Read motors speed
            controller_speeds = speeds_reader.read_data()
            # Move motors
            speeds = motors_control.move_and_read(controller_speeds)
            print(speeds)
            # Process to get motors speed

    except KeyboardInterrupt:
        print("Ctrl-C pressed!")
        sys.exit(0)


if __name__ == "__main__":
    print("prev func main")
    main()
