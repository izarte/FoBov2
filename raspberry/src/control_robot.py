# from motors_control import MotorsControl
import tkinter as tk
from websockets.sync.client import connect
import json


def on_key_press(event, label):
    speed = 10
    w = 0
    left_speed = 0
    right_speed = 0
    # motors_control = MotorsControl()

    key = event.keysym.lower()
    if key == "p":
        root.destroy()  # This will close the GUI and terminate the program
    if key == "w":
        speed = speed
    elif key == "s":
        speed = -speed
    if key == "i":
        speed += 10
        if speed >= 100:
            speed = 100
    elif key == "k":
        speed -= 10
        if speed <= 0:
            speed = 0
    if key == "a":
        w = -1
    elif key == "d":
        w = 1
    if key == "c":
        speed = 0
        w = 0

    left_speed = speed
    right_speed = speed
    if w > 0:
        right_speed = speed
        left_speed = -speed
    elif w < 0:
        right_speed = -speed
        left_speed = speed
    data = {"speeds": (left_speed, right_speed)}
    with connect("ws://192.168.1.254:8002/") as ws:
        ws.send(json.dumps(data))
        ws.close()

    # motors_control.move_and_read((left_speed, right_speed))


print(
    "Press 'w' to move forward, 's' to move backwards, 'a' to turn left, 'd' to turn right, and 'p' to exit."
)
root = tk.Tk()
root.title("Robot Speed Controller")
label = tk.Label(
    root,
    text="Press:\n'w' to move forward\n's' to move backwards\n'a' to turn left\n'd' to turn right\n'i' to increase speed\n'k' to decrease speed'p' to exit.",
)
label.pack(pady=20)

# Bind key events to the on_key_press function
root.bind("<KeyPress>", lambda event: on_key_press(event, label))

root.mainloop()
