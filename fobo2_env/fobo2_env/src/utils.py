import cv2
import numpy as np
import pybullet as p


def random_pos_orientation(area):
    startPos = [
        np.random.uniform(area[0][0], area[0][1]),
        np.random.uniform(area[1][0], area[1][1]),
        area[2],
    ]
    startOrientation = p.getQuaternionFromEuler(
        [0, 0, np.random.uniform(-np.pi, np.pi)]
    )

    return startPos, startOrientation


def get_human_robot_distance(client_id: int, robot_id: int, human_id: int) -> float:
    robot_position, _ = p.getBasePositionAndOrientation(
        physicsClientId=client_id, bodyUniqueId=robot_id
    )
    human_position, _ = p.getBasePositionAndOrientation(
        physicsClientId=client_id, bodyUniqueId=human_id
    )
    distance = np.sqrt(
        np.sum(
            (
                np.array([robot_position[0], robot_position[1]])
                - np.array([human_position[0], human_position[1]])
            )
            ** 2
        )
    )
    return distance


def rotate_by_yaw(position: np.array, yaw: float):
    r_pos = list(position)
    yaw = np.radians(yaw)
    r_pos[0] = -np.sin(yaw) * position[0]
    r_pos[1] = np.cos(yaw) * position[0]
    return np.array(r_pos)


def orientation_in_grad(orientation):
    roll = np.degrees(orientation[0])
    pitch = np.degrees(orientation[1])
    yaw = np.degrees(orientation[2])

    return roll, pitch, yaw + 90


def get_human_coordinates(rgb):
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([30, 30, 30])

    # Create a mask to isolate human as black region
    mask = cv2.inRange(rgb, lower_black, upper_black)
    # cv2.imwrite("test.png", mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centroid = (-1, -1)
    # Check if contours are found
    if contours:
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        # Calculate the moments of the contour
        M = cv2.moments(largest_contour)
        if M["m00"]:
            # Calculate the centroid
            centroid_x = int(M["m10"] / M["m00"])
            centroid_y = int(M["m01"] / M["m00"])
            centroid = (centroid_x, centroid_y)
    return centroid


def value_in_range(value: float, center: float, offset: float):
    difference = np.abs(value - center)
    if difference <= offset:
        return True
    return False

def position_in_range(position, area):
    if position[0] < area[0][0] or position[0] > area[0][1] or position[1] < area[1][0] or position[1] > area[1][1]:
        return False
    return True

def add_noise_to_motors(speeds, max_value):
    std = 0.01
    noise_speeds = np.random.normal(loc=0, scale=std, size=2).astype(np.float64)
    speeds += noise_speeds

    for i in [0, 1]:
        if speeds[i] > max_value:
            speeds[i] = max_value
        elif speeds[i] < -max_value:
            speeds[i] = -max_value
    return speeds