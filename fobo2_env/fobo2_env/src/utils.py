import pybullet as p
import numpy as np
import cv2


def random_pos_orientation(area):
    startPos = [
        np.random.uniform(area[0][0], area[0][1]),
        np.random.uniform(area[1][0], area[1][1]),
        area[2]
    ]
    startOrientation = p.getQuaternionFromEuler([0,0,np.random.uniform(-np.pi, np.pi)])

    return startPos, startOrientation


def distance_in_range(distance : float, desired_distance : float, offset : float) -> bool:
    difference = distance - desired_distance
    if difference > 0 and difference < offset:
        return True
    if difference < 0 and difference < - offset:
        return True
    return False


def get_human_robot_distance(client_id : int, robot_id : int, human_id : int) -> float:

    robot_position, _ =  p.getBasePositionAndOrientation(
        physicsClientId = client_id,
        bodyUniqueId = robot_id
        )
    human_position, _ =  p.getBasePositionAndOrientation(
        physicsClientId = client_id,
        bodyUniqueId = human_id
        )
    distance = np.sqrt(
        np.sum(
            (np.array([robot_position[0], robot_position[1]]) - 
                np.array([human_position[0], human_position[1]]))**2
        )
    )
    return distance

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
        
        # Calculate the centroid
        centroid_x = int(M['m10'] / M['m00'])
        centroid_y = int(M['m01'] / M['m00'])
        centroid = (centroid_x, centroid_y)
    return centroid
        