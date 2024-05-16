import serial
import time


class MotorsControl:
    def __init__(self):
        self.ser = serial.Serial("/dev/serial0", 1500000, timeout=1)
        self.ser.flush()

    def move_and_read(self, speed: tuple[float, float]) -> tuple[float, float]:
        speed_str = f"{speed[0]} {speed[1]}"
        print("Sending: ", speed_str)
        self.ser.write(speed_str.encode("utf_8"))
        speeds_str = ""
        while not speeds_str:
            speeds_str = self.ser.readline().decode("utf-8")
        speeds = speeds_str.split(" ")

        return [float(speeds[0]), float(speeds[1])]


def main():
    while True:
        string = "0.3 0"  # input from user
        print("sending ", string)
        string = string.encode("utf_8")
        t1 = time.time_ns()
        ser.write(string)
        # time.sleep(1)  # delay of 1 second
        line = ""
        while not line:
            line = ser.readline().decode("utf-8")
        t2 = time.time_ns()
        print(f"received: {line}")
        print(f"in {t2 - t1} ns, {(t2 - t1) / 1000000000} s")


if __name__ == "__main__":
    ser = serial.Serial("/dev/serial0", 1500000, timeout=1)
    ser.flush()
    main()
