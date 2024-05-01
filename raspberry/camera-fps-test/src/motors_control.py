import serial
import time


class MotorsControl:
    def __init__(self):
        self.ser = serial.Serial("/dev/ttyS0", 115200, timeout=1)
        self.ser.flush()

    def move(self, speed: tuple[float, float]):
        speed_str = f"{speed[0]} {speed[1]}"
        self.ser.write(speed_str)

    def read_encoders(self) -> tuple[float, float]:
        self.ser.write("Read")
        speeds_str = self.ser.readline().decode("utf-8").rstrip()
        speeds = speeds_str.split(" ")

        return [float(speeds[0]), float(speeds[1])]


def main():
    while True:
        string = input("enter string:")  # input from user
        string = string + "\n"  # "\n" for line seperation
        string = string.encode("utf_8")
        ser.write(string)
        line = ser.readline().decode("utf-8").rstrip()
        print("received: ", line)
        time.sleep(1)  # delay of 1 second


if __name__ == "__main__":
    # if connected via serial Pin(RX, TX)
    ser = serial.Serial(
        "/dev/ttyS0", 9600, timeout=1
    )  # 9600 is baud rate(must be same with that of NodeMCU)
    ser.flush()
