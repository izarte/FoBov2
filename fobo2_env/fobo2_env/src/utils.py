import pybullet as p

# TODO return random position and orientation, maybe between some limits given
def random_pos_orientation():
    startPos = [0,0,1.1]
    startOrientation = p.getQuaternionFromEuler([0,0,0])

    return startPos, startOrientation


def distance_in_range(distance : float, desired_distance : float, offset : float) -> bool:
    difference = distance - desired_distance
    if difference > 0 and difference < offset:
        return True
    if difference < 0 and difference < - offset:
        return True
    return False
