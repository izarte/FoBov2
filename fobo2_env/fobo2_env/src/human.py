import os
import pybullet as p
import numpy as np

from fobo2_env.src.utils import (
    random_pos_orientation,
    rotate_by_yaw,
    value_in_range,
)
from fobo2_env.src.walking_simulation_utils import (
    get_walking_sim_uniform_joints_positions,
)


class Human:
    def __init__(self, client_id):
        self.client_id = client_id
        self.id = 0
        self.joints_id = {
            "right_hip": 33,
            "right_knee": 35,
            "right_ankle": 38,
            "left_hip": 42,
            "left_knee": 44,
            "left_ankle": 47,
        }
        self.total_step_division = 200
        self.joints_simulation = {}
        self.current_left_simulation = 0
        self.current_right_simulation = 0
        self.step_base_movement_array = np.array([0.005, 0, 0])
        self.start_pose = 0

    def reset(self, starting_area):
        self.starting_area = starting_area
        # Generate random human start position
        human_start_pos, _ = random_pos_orientation(
            (starting_area[0], starting_area[1], 1.1)
        )

        human_speed = np.random.randint(5, 10)
        self.total_step_division = int(5 / 2 * 100)
        self.step_base_movement_array = np.array([0.001 * human_speed, 0, 0])

        uniform_ankle, uniform_knee, uniform_hip = (
            get_walking_sim_uniform_joints_positions(self.total_step_division)
        )
        self.joints_simulation = {
            "ankle": np.radians(uniform_ankle),
            "knee": np.radians(uniform_knee),
            "hip": np.radians(uniform_hip),
        }
        self.current_left_simulation = self.total_step_division
        self.current_right_simulation = self.total_step_division // 2

        # Generate random target for walking and get human orientation
        human_quaternion = self.generate_new_target(position=human_start_pos)

        self.id = p.loadURDF(
            physicsClientId=self.client_id,
            fileName=os.path.dirname(__file__) + "/models/human.urdf",
            basePosition=human_start_pos,
            baseOrientation=human_quaternion,
            useFixedBase=False,
        )
        
        self.start_pose = human_start_pos

        # Remove default pybullet mass in human
        for i in range(
            p.getNumJoints(physicsClientId=self.client_id, bodyUniqueId=self.id)
        ):
            if i < 32:
                p.changeDynamics(
                    physicsClientId=self.client_id,
                    bodyUniqueId=self.id,
                    linkIndex=i,
                    mass=0,
                )
        p.changeDynamics(
            physicsClientId=self.client_id, bodyUniqueId=self.id, linkIndex=-1, mass=0
        )
        self.current_simulation_step = 0

    def step(self):
        position, quaternion = p.getBasePositionAndOrientation(bodyUniqueId=self.id)
        # Move human to target direction
        new_position = np.array(position) + self.step_movement_array
        # Check if human is near target
        if self.check_target(position=new_position):
            # Generate new random target and get corresponding quaterion orientation
            quaternion = self.generate_new_target(position=new_position)

        p.resetBasePositionAndOrientation(
            bodyUniqueId=self.id, posObj=new_position, ornObj=quaternion
        )
        p.setJointMotorControl2(
            bodyIndex=self.id,
            jointIndex=self.joints_id["right_hip"],
            controlMode=p.POSITION_CONTROL,
            targetPosition=self.joints_simulation["hip"][self.current_right_simulation],
        )
        p.setJointMotorControl2(
            bodyIndex=self.id,
            jointIndex=self.joints_id["right_knee"],
            controlMode=p.POSITION_CONTROL,
            targetPosition=self.joints_simulation["knee"][
                self.current_right_simulation
            ],
        )
        p.setJointMotorControl2(
            bodyIndex=self.id,
            jointIndex=self.joints_id["right_ankle"],
            controlMode=p.POSITION_CONTROL,
            targetPosition=self.joints_simulation["ankle"][
                self.current_right_simulation
            ],
        )

        p.setJointMotorControl2(
            bodyIndex=self.id,
            jointIndex=self.joints_id["left_hip"],
            controlMode=p.POSITION_CONTROL,
            targetPosition=self.joints_simulation["hip"][self.current_left_simulation],
        )
        p.setJointMotorControl2(
            bodyIndex=self.id,
            jointIndex=self.joints_id["left_knee"],
            controlMode=p.POSITION_CONTROL,
            targetPosition=self.joints_simulation["knee"][self.current_left_simulation],
        )
        p.setJointMotorControl2(
            bodyIndex=self.id,
            jointIndex=self.joints_id["left_ankle"],
            controlMode=p.POSITION_CONTROL,
            targetPosition=self.joints_simulation["ankle"][
                self.current_left_simulation
            ],
        )

        self.current_left_simulation -= 1
        self.current_right_simulation -= 1
        if self.current_left_simulation == -1:
            self.current_left_simulation = self.total_step_division
        if self.current_right_simulation == -1:
            self.current_right_simulation = self.total_step_division

    def check_target(self, position):
        distance = np.sqrt(
            np.sum(
                (
                    np.array([position[0], position[1]])
                    - np.array([self.target[0], self.target[1]])
                )
                ** 2
            )
        )
        return value_in_range(value=distance, center=0.1, offset=0.05)

    def generate_new_target(self, position):
        # Get new random position for target
        self.target, _ = random_pos_orientation(
            (self.starting_area[0], self.starting_area[1], 1.1)
        )
        # Calculate direction between human and target
        yaw = np.arctan2(
            self.target[1] - position[1],
            self.target[0] - position[0],
        )
        # Create quaternion pose for human
        human_quaternion = p.getQuaternionFromEuler([0, 0, yaw])
        # Rotate movement array by calculated yaw with 90 degrees offset
        self.step_movement_array = rotate_by_yaw(
            self.step_base_movement_array, np.degrees(yaw) - 90
        )
        return human_quaternion
