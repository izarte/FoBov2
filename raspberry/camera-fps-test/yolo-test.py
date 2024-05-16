from pycoral.utils.edgetpu import list_edge_tpus, make_interpreter
from pycoral.adapters.common import input_size
from pycoral.utils import dataset
import cv2
import numpy as np

# List available Edge TPUs
print(list_edge_tpus())

# Load the model
interpreter = make_interpreter("yolov8n_float32_edgetpu.tflite")
interpreter.allocate_tensors()

# Get input details to understand the preprocessing required
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height, width = input_size(interpreter)

# Load and preprocess the image
img = cv2.imread("foto.jpg")
resized_img = cv2.resize(img, (width, height))

# Check if the model requires normalized input
if input_details[0]['dtype'] == np.float32:
    input_data = np.expand_dims(resized_img / 255.0, axis=0).astype(np.float32)
else:
    input_data = np.expand_dims(resized_img, axis=0).astype(input_details[0]['dtype'])

# Set the input tensor
interpreter.set_tensor(input_details[0]['index'], input_data)

# Run inference
interpreter.invoke()

# Get the output tensor
output_data = interpreter.get_tensor(output_details[0]['index'])

print("Output data:", output_data)
