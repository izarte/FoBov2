import pybullet as p
import numpy as np

# TODO return random position and orientation, maybe between some limits given
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
