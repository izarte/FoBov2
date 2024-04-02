import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data

from fobo2_env.src.utils import random_pos_orientation
from fobo2_env.src.robot_fobo import Robot
class FoBo2Env(gym.Env):
    metadata = {'render_modes': ['DIRECT', 'GUI'], 'render_fps': 60}  
    def __init__(self, render_mode="DIRECT", rgb_width = 320, rgb_height = 320, depth_width = 320, depth_height= 320):
        # Observation space are input variables for system

        self.observation_space = spaces.Dict(
            {
                "wheels-speed": spaces.Box(low=np.array([-100, -100]), high=np.array([100, 100]), dtype=np.float32),
                "human-pixel": spaces.Box(low=np.array([0, 0]), high=np.array([rgb_width-1, rgb_height-1]), dtype=np.float32),
                "depth-image": spaces.Box(low=0, high=255, shape=(depth_width, depth_height, 1), dtype=np.uint8)
                # "desired-distance": spaces.Box(low=-0.1, high=4, shape=(1,),  dtype=float),
            }
        )
        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Box(low=np.array([-100, -100]), high=np.array([100, 100]), dtype=np.float32)
        
        self.depth_width = depth_width
        self.depth_height = depth_height
        self.rgb_width = rgb_width
        self.rgb_height = rgb_height



        self.depth_camera_realtive_position = [0.1, 0, 0.5]
        self.rgb_camera_realtive_position = [-0.1, 0, 0.5]

        if render_mode == "DIRECT":
            self._client = p.connect(p.DIRECT)
        elif render_mode == "GUI":
            self._client = p.connect(p.GUI)
        

        # Create variable placeholders for pybullet objects
        self._planeId = None
        self._human_id = None
        self._robot_id = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Reset simulation
        p.resetSimulation(self._client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) 
        p.setGravity(0, 0, -9.81, self._client)

        self._planeId = p.loadURDF(
            physicsClientId = self._client,
            fileName="plane.urdf", 
            useFixedBase = True
        )

        human_start_pos, human_start_orientation = random_pos_orientation()
        robot_start_pos, robot_start_orientation = random_pos_orientation()

        self._human_id = p.loadURDF(
            physicsClientId = self._client,
            fileName="urdf/human.urdf", 
            basePosition = [1, 0, 1],
            baseOrientation = human_start_orientation,
            useFixedBase = False
        )

        # Remove default pybullet mass in human 
        for i in range(p.getNumJoints(physicsClientId = self._client, bodyUniqueId = self._human_id )):
            if i < 32:
                p.changeDynamics(physicsClientId = self._client, bodyUniqueId = self._human_id, linkIndex = i, mass = 0)
        p.changeDynamics(physicsClientId = self._client, bodyUniqueId = self._human_id, linkIndex = -1, mass = 0)

        self.robot = Robot(
            client_id = self._client,
            depth_width = self.depth_width,
            depth_height = self.depth_height,
            rgb_width = self.rgb_width,
            rgb_height = self.rgb_height
        )
        self._robot_id = self.robot.reset()
        # self.robot_id = p.loadURDF(
        #     physicsClientId = self._client,
        #     fileName="urdf/fobo2.urdf", 
        #     basePosition = robot_start_pos,
        #     baseOrientation = robot_start_orientation,
        #     useFixedBase = False
        # )

        observation = self._get_observation()
        info = self._get_info()

        return observation, info
    
    def step(self, action):
        p.stepSimulation(physicsClientId = self._client)
        self._human_walk()
        self._move_robot(action)
        observation = self._get_observation()

        # p.performCollisionDetection(physicsClientId = self._client)
        # contact_points = p.getContactPoints(physicsClientId = self._client, bodyA = self._robot_id)

        reward = self._compute_reward()
        terminated, truncated = self._get_end_episode()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def close(self):
        p.disconnect(self._client) 

    # TODO human walking movement and animation 
    def _human_walk(self):
        return
    
    # TODO give robot wheels speed with action given, maybe camera should move here
    def _move_robot(self, action):
        # Set action speed in both wheels
        self.robot.move(action=action)

        return 

    # TODO calculate reward in this moment, should be based on robot-human distance
    def _compute_reward(self):
        return
    
    # TODO check if episode es terminated, human reached target point, or truncated if any collision has been detected
    def _get_end_episode(self):
        return 0, 0

    # TODO read robot observations, speeds, images
    def _get_observation(self):
        rgb, depth = self.robot.get_images()
        return {"wheels-speed": 0, "human-pixel": [0, 0], "depth-image": 0}

    # TODO get current environment information
    def _get_info(self):
        return {}


