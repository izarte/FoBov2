# import cv2
# import time
# import numpy as np
# from multiprocessing import Process
# from multiprocessing import Queue
# from ultralytics import YOLO

# from PIL import Image


# CONF_THRESHOLD = 0.8

# # define the function that handles our processing thread
# def process_video(model_path:str, video_source, show:bool=True):
#     queuepulls = 0.0
#     detections = 0
#     # init video
#     print("init video")
#     cap = cv2.VideoCapture(video_source)

#     frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     img = None
#     out = None
#     print("create model")
#     model = YOLO(model_path,task="detect")

#     # time the frame rate....
#     frames = 0
#     start_time = time.time()
#     print("start loop")
#     while (frames < 1000):
#         # Capture frame-by-frame
#         ret, frame = cap.read()
#         print(frames)
#         frames += 1
#         if ret == True:
#             # Capture frame-by-frame
#             # frame = frame.array
#             img = Image.fromarray(frame)
#             objs = model.predict(img,conf=CONF_THRESHOLD,verbose=False)[0]

#             if out is not None:
#                 # loop over the detections
#                 for box  in out:
#                     xmin = int(box[0])
#                     ymin = int(box[1])
#                     xmax = int(box[2])
#                     ymax = int(box[3])
#                     objID = int(box[5])
#                     confidence = box[4]

#                     if confidence > CONF_THRESHOLD:
#                         # bounding box
#                         cv2.rectangle(frame, (xmin, ymin),
#                                     (xmax, ymax), color=(0, 0, 255))
#                         detections += 1  # positive detections
#                 queuepulls += 1

#             if show:
#                 # Display the resulting frame
#                 cv2.rectangle(frame, (0, 0),
#                             (frameWidth, 20), (0, 0, 0), -1)

#                 cv2.rectangle(frame, (0, frameHeight-20),
#                             (frameWidth, frameHeight), (0, 0, 0), -1)
#                 cv2.putText(frame, 'Threshold: '+str(round(CONF_THRESHOLD, 1)), (10, 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1, cv2.LINE_AA)

#                 cv2.putText(frame, 'Positive detections: '+str(detections), (10, frameHeight-10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1, cv2.LINE_AA)


#                 cv2.namedWindow('Coral', cv2.WINDOW_NORMAL)
#                 cv2.resizeWindow('Coral', frameWidth, frameHeight)
#                 cv2.imshow('Coral', frame)
#                 cv2.waitKey(1)

#         # Break the loop
#         else:
#             break

#     elapsed_time = time.time() - start_time
#     print("elapsed time: ", elapsed_time)
#     print("fps:", frames / elapsed_time)
#     # Everything done, release the vid
#     cap.release()

#     cv2.destroyAllWindows()

# import configparser

# if __name__ == "__main__":
#     modelPath = "yolov8n_float32.tflite"
#     camera_idx = 0
#     show = False
#     print("MAIN")
#     process_video(model_path=modelPath,video_source=camera_idx,show=show)


import cv2
import time
import numpy as np
from multiprocessing import Process
from multiprocessing import Queue
from ultralytics import YOLO

from PIL import Image
import argparse


# define the function that handles our processing thread
def process_video(
    model_path: str,
    video_source,
    pwm_gpio: int,
    show: bool = True,
    enable_motor: bool = False,
):
    global model
    motor_pos = [0, 45, 90, 135, 180]
    motor_index = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    queuepulls = 0.0
    detections = 0
    fps = 0.0
    qfps = 0.0
    # init video
    cap = cv2.VideoCapture(video_source)

    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # initialize the input queue (frames), output queue (out),
    # and the list of actual detections returned by the child process
    inputQueue = Queue(maxsize=1)
    outputQueue = Queue(maxsize=1)
    img = None
    out = None
    model = YOLO(model_path, task="detect")

    # construct a child process *indepedent* from our main process of
    # execution
    p = Process(
        target=classify_frame,
        args=(
            img,
            inputQueue,
            outputQueue,
        ),
    )
    p.daemon = True
    p.start()
    time.sleep(10)

    # time the frame rate....
    timer1 = time.time()
    frames = 0
    queuepulls = 0
    timer2 = 0
    t2secs = 0
    process = True
    while cap.isOpened() and process:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret:
            if queuepulls == 1:
                timer2 = time.time()
            # Capture frame-by-frame
            # frame = frame.array
            # img = Image.fromarray(frame)
            # if the input queue *is* empty, give the current frame to
            # classify
            if inputQueue.empty():
                inputQueue.put(frame)

            # if the output queue *is not* empty, grab the detections
            if not outputQueue.empty():
                out = outputQueue.get()

            data = {"pixel": {"x": 0, "y": 0}}
            if out is not None:
                if out.id is not None:
                    boxes = out
                    print(boxes)
                    print(boxes)
                    x = int(
                        boxes.xyxy[0][0]
                        + ((boxes.xyxy[0][2] - boxes.xyxy[0][0]).item() / 2)
                    )
                    y = int(
                        boxes.xyxy[0][1]
                        + ((boxes.xyxy[0][3] - boxes.xyxy[0][1]).item() / 2)
                    )
                    # print(x, y)
                    data["pixel"]["x"] = x
                    data["pixel"]["y"] = y
                    # FPS calculation
                    frames += 1
                    if frames % 1000 == 0:
                        end1 = time.time()
                        t1secs = end1 - timer1
                        fps = round(frames / t1secs, 2)
                        print("FPS: ", fps)
                    if queuepulls > 1:
                        end2 = time.time()
                        t2secs = end2 - timer2
                        qfps = round(queuepulls / t2secs, 2)
                else:
                    data["pixel"] = {"x": -1, "y": -1}
                out = None
                print(data)

        # Break the loop
        else:
            break

    p.join()
    # Everything done, release the vid
    cap.release()

    cv2.destroyAllWindows()


def classify_frame(img, inputQueue, outputQueue):
    global model
    global confThreshold
    while True:
        # check to see if there is a frame in our input queue
        if not inputQueue.empty():
            # grab the frame from the input queue
            img = inputQueue.get()
            img = Image.fromarray(img)
            img = img.resize((320, 320), Image.ANTIALIAS)
            t1 = time.time()
            objs = model.predict(img, conf=confThreshold, classes=0, verbose=False)[0]
            print("inference time: ", time.time() - t1)
            outputQueue.put(objs.boxes.cpu())


import configparser

if __name__ == "__main__":
    modelPath = "/usr/src/camera-test/yolov8n_float32_edgetpu.tflite"
    camera_idx = 1
    confThreshold = 0.5
    pwm_gpio = 1
    show = False
    enable_motor = False
    process_video(
        model_path=modelPath,
        video_source=camera_idx,
        pwm_gpio=pwm_gpio,
        show=show,
        enable_motor=enable_motor,
    )
