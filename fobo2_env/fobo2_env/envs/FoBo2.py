import gymnasium as gym
import numpy as np
import pyb_utils
import pybullet as p
import pybullet_data
from gymnasium import spaces

from fobo2_env.src.human import Human
from fobo2_env.src.robot_fobo import Robot
from fobo2_env.src.utils import (
    distance_in_range,
    get_human_coordinates,
    get_human_robot_distance,
)
from fobo2_env.src.world_generation import World


class FoBo2Env(gym.Env):
    metadata = {"render_modes": ["DIRECT", "GUI"], "render_fps": 60}

    def __init__(
        self,
        render_mode="DIRECT",
        rgb_width=320,
        rgb_height=320,
        depth_width=320,
        depth_height=320,
    ):
        # Observation space are input variables for system

        self.observation_space = spaces.Dict(
            {
                "wheels-speed": spaces.Box(
                    low=np.array([-100, -100]),
                    high=np.array([100, 100]),
                    dtype=np.float32,
                ),
                "human-pixel": spaces.Box(
                    low=0, high=255, shape=(rgb_width, rgb_height, 3), dtype=np.uint8
                ),
                "depth-image": spaces.Box(
                    low=0,
                    high=255,
                    shape=(depth_width, depth_height, 1),
                    dtype=np.uint8,
                ),
                # "desired-distance": spaces.Box(low=-0.1, high=4, shape=(1,),  dtype=float),
            }
        )
        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Box(
            low=np.array([-100, -100]), high=np.array([100, 100]), dtype=np.float32
        )

        self.depth_width = depth_width
        self.depth_height = depth_height
        self.rgb_width = rgb_width
        self.rgb_height = rgb_height

        self.depth_camera_realtive_position = [0.1, 0, 0.5]
        self.rgb_camera_realtive_position = [-0.1, 0, 0.5]
        self.desired_distance = 1.5
        self.offset = 0.5

        if render_mode == "DIRECT":
            self._client_id = p.connect(p.DIRECT)
        elif render_mode == "GUI":
            self._client_id = p.connect(p.GUI)

        # Create variable placeholders for pybullet objects
        self._planeId = None
        self._human_id = None
        self._robot = None
        self._world = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Reset simulation
        p.resetSimulation(self._client_id)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81, self._client_id)

        self._planeId = p.loadURDF(
            physicsClientId=self._client_id, fileName="plane.urdf", useFixedBase=True
        )
        # Create world object
        self._world = World(self._client_id)
        # Generate basic room, 4 walls
        self._world.create_basic_room()

        # Get spawning zone for robot and human percentage
        robot_area, human_area = self._world.calculate_starting_areas(area1=1, area2=1)

        self._human = Human(self._client_id)
        self._human.reset(starting_area=human_area)

        self._robot = Robot(
            client_id=self._client_id,
            depth_width=self.depth_width,
            depth_height=self.depth_height,
            rgb_width=self.rgb_width,
            rgb_height=self.rgb_height,
        )
        distance = 0
        while distance < 0.8:
            self._robot.reset(starting_area=robot_area)
            distance = get_human_robot_distance(
                client_id=self._client_id,
                robot_id=self._robot.id,
                human_id=self._human.id,
            )

        # self.robot_id = p.loadURDF(
        #     physicsClientId = self._client_id,
        #     fileName="urdf/fobo2.urdf",
        #     basePosition = robot_start_pos,
        #     baseOrientation = robot_start_orientation,
        #     useFixedBase = False
        # )

        observation = self._get_observation()
        info = self._get_info()

        self.relevant_collisions = [(self._robot.id, self._human.id)]
        for wall_id in self._world.ids:
            self.relevant_collisions.append((self._robot.id, wall_id))

        self.collision_detector = pyb_utils.CollisionDetector(
            client_id=self._client_id, collision_pairs=self.relevant_collisions
        )

        return observation, info

    def step(self, action):
        p.stepSimulation(physicsClientId=self._client_id)
        self._human_walk()
        self._move_robot(action)
        observation = self._get_observation()
        reward = self._compute_reward(observation)
        terminated, truncated = self._get_end_episode()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def close(self):
        p.disconnect(self._client_id)

    # TODO human walking movement and animation
    def _human_walk(self):
        return

    def _move_robot(self, action):
        # Set action speed in both wheels
        self._robot.move(action=action)

        return

    def _compute_reward(self, observation):
        distance = get_human_robot_distance(
            client_id=self._client_id, robot_id=self._robot.id, human_id=self._human.id
        )
        reward = 0.0
        if distance_in_range(
            distance=distance,
            desired_distance=self.desired_distance,
            offset=self.offset,
        ):
            reward += 1.0

        if observation["human-pixel"][0] >= 0:
            reward += 1.0

        return reward

    def _get_end_episode(self):
        truncated = False
        terminated = False
        if self.collision_detector.in_collision(margin=0.0):
            truncated = True
        return terminated, truncated

    def _get_observation(self):
        observations = {"wheels-speed": 0, "human-pixel": [0, 0], "depth-image": 0}
        rgb, depth = self._robot.get_images()
        x, y = get_human_coordinates(rgb)
        observations["depth-image"] = depth
        observations["human-pixel"] = np.array([x, y], dtype=np.uint8)
        speedL, speedR = self._robot.get_motor_speeds()
        observations["wheels-speed"] = np.array([speedL, speedR], dtype=np.float32)
        return observations

    # TODO get current environment information
    def _get_info(self):
        return {}
