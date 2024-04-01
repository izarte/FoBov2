import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p

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

        self.depth_camera_params = {
            "fov": 60,
            "near": 0.02,
            "far": 1,
            "aspect": self.depth_width / self.depth_height
        }

        self.rgb_camera_params = {
            "fov": 60,
            "near": 0.02,
            "far": 1,
            "aspect": self.rgb_width / self.rgb_height
        }

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
        p.setGravity(0, 0, -9.81, self._client)

        self._planeId = p.loadURDF("plane.urdf", )

        self._planeId = p.loadURDF(
            physicsClientId = self._client,
            fileName="plane.urdf", 
            useFixedBase = True
        )

        human_start_pos, human_start_orientation = self._random_pos_orientation()
        robot_start_pos, robot_start_orientation = self._random_pos_orientation()

        self._human_id = p.loadURDF(
            physicsClientId = self._client,
            fileName="urdf/human.urdf", 
            basePosition = human_start_pos,
            baseOrientation = human_start_orientation,
            useFixedBase = False
        )
        self.robot_id = p.loadURDF(
            physicsClientId = self._client,
            fileName="urdf/fobo2.urdf", 
            basePosition = robot_start_pos,
            baseOrientation = robot_start_orientation,
            useFixedBase = False
        )
    
    def step(self, action):
        self._human_walk()
        self._move_robot(action)

        p.stepSimulation(physicsClientId = self._client)
        p.performCollisionDetection(physicsClientId = self._client)
        contact_points = p.getContactPoints(physicsClientId = self._client, bodyA = self._robot_id)

        reward = self._compute_reward()
        terminated, truncated = self._get_end_episode()
        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def close(self):
        p.disconnect(self._client) 

    # TODO human walking movement and animation 
    def _human_walk():
        return
    
    # TODO give robot wheels speed with action given, maybe camera should move here
    def _move_robot(action):
        return 

    # TODO calculate reward in this moment, should be based on robot-human distance
    def _compute_reward():
        return
    
    # TODO check if episode es terminated, human reached target point, or truncated if any collision has been detected
    def _get_end_episode():
        return

    # TODO read robot observations, speeds, images
    def _get_observation():
        return

    # TODO get current environment information
    def _get_info():
        return

    # TODO return random position and orientation, maybe between some limits given
    def _random_pos_orientation():
        startPos = [0,0,1.1]
        startOrientation = p.getQuaternionFromEuler([0,0,0])

        return startPos, startOrientation
