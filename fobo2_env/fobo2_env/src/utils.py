import pybullet as p

# TODO return random position and orientation, maybe between some limits given
def random_pos_orientation():
    startPos = [0,0,1.1]
    startOrientation = p.getQuaternionFromEuler([0,0,0])

    return startPos, startOrientation