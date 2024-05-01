from data_reader import DataReader
from motors_control import MotorsControl
import sys


def main():
    # Creater reader objects
    yolo_reader = DataReader(port=8000, label="pixel")
    depth_reader = DataReader(port=8001, label="depth")
    motors_control = MotorsControl()

    try:
        while True:
            # Read yolo data
            human_pixel = yolo_reader.read_data()
            # Read depth data
            depth_image = depth_reader.read_data()
            # Read motors speed
            speeds = motors_control.read_encoders()
            # Process to get motors speed

            # Move motors
            motors_control.move(speeds)
    except KeyboardInterrupt:
        print("Ctrl-C pressed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
