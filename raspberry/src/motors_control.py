import serial
import time


class MotorsControl:
    def __init__(self):
        self.ser = serial.Serial("/dev/serial0", 1500000, timeout=1)
        self.ser.flush()

    def move_and_read(self, speed: tuple[float, float]) -> tuple[float, float]:
        speed_str = f"{speed[0]} {speed[1]}\n"
        print("Sending: ", speed_str)
        speed_str = speed_str.encode("utf_8")
        self.ser.write(speed_str)
        speeds_str_input = ""
        left_speed = 0
        right_speed = 0
        while not speeds_str_input:
            try:
                speeds_str_input = self.ser.readline().decode("utf-8")
            except:  # noqa: E722
                speeds_str_input = None
                self.ser.write(speed_str)
            if not speeds_str_input:
                speeds_str_input = None
                self.ser.write(speed_str)
                continue
            try:
                speeds = speeds_str_input.split(" ")
                left_speed = float(speeds[0])
                right_speed = float(speeds[1])
            except:  # noqa: E722
                speeds_str_input = None
                self.ser.write(speed_str)

        return [left_speed, right_speed]


def main():
    while True:
        string = "0.3 0\n"  # input from user
        print("sending ", string)
        string = string.encode("utf_8")
        t1 = time.time_ns()
        ser.write(string)
        # time.sleep(1)  # delay of 1 second
        line = ""
        while not line:
            try:
                line = ser.readline().decode("utf-8")
            except:  # noqa: E722
                line = None
                ser.write(string)
        t2 = time.time_ns()
        print(f"received: {line}")
        print(f"in {t2 - t1} ns, {(t2 - t1) / 1000000000} s")


if __name__ == "__main__":
    ser = serial.Serial("/dev/serial0", 1500000, timeout=1)
    ser.flush()
    main()
