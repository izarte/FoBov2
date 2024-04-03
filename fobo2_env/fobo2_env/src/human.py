import pybullet as p
import os

from fobo2_env.src.utils import random_pos_orientation

class Human():
    def __init__(self, client_id):
        self.client_id = client_id
        self.id = 0

    def reset(self, starting_area):
        human_start_pos, human_start_orientation = random_pos_orientation((starting_area[0], starting_area[1], 1.1))
        print(human_start_pos)
        self.id = p.loadURDF(
            physicsClientId = self.client_id,
            fileName = os.path.dirname(__file__) + "/models/human.urdf", 
            basePosition = human_start_pos,
            baseOrientation = human_start_orientation,
            useFixedBase = False
        )

        # Remove default pybullet mass in human 
        for i in range(p.getNumJoints(physicsClientId = self.client_id, bodyUniqueId = self.id )):
            if i < 32:
                p.changeDynamics(physicsClientId = self.client_id, bodyUniqueId = self.id, linkIndex = i, mass = 0)
        p.changeDynamics(physicsClientId = self.client_id, bodyUniqueId = self.id, linkIndex = -1, mass = 0)

