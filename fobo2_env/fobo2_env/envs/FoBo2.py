import gymnasium as gym
import numpy as np
import pyb_utils
import pybullet as p
import pybullet_data
from gymnasium import spaces

from fobo2_env.src.observation_manager import ObservationManager
from fobo2_env.src.robot_tracker import RobotTracker
from fobo2_env.src.human import Human
from fobo2_env.src.robot_fobo import Robot
from fobo2_env.src.reward_utils import calculate_distance_reward, calculate_pixel_reward
from fobo2_env.src.utils import get_human_coordinates, get_human_robot_distance
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
        memory=4,
    ):
        # Define internal variables
        self.memory = memory
        self.depth_width = depth_width
        self.depth_height = depth_height
        self.rgb_width = rgb_width
        self.rgb_height = rgb_height

        # Define constant values
        self.desired_distance = 1.5
        self.offset = 0.2
        self.start_scoring_offset = 3
        self.max_track = 1000
        self.start_terminated_check = 1000
        self.max_steps = 5000

        self.boundings = {
            "wheels_speed": [-35, 35],
            "human_pixel": [0, max(rgb_height, rgb_width)],
            "depth_image": [0, 255],
        }
        self.dtypes = {
            "wheels_speed": np.float32,
            "human_pixel": np.float32,
            "depth_image": np.uint8,
        }
        self.observation_manager = ObservationManager(
            memory=memory, boundings=self.boundings, dtypes=self.dtypes
        )
        # Observation space are input variables for system
        # All values are normalized in observation manager
        self.observation_space = spaces.Dict(
            {
                "wheels_speed": spaces.Box(
                    low=-1,
                    high=1,
                    shape=(memory * 2,),
                    dtype=self.dtypes["wheels_speed"],
                ),
                "human_pixel": spaces.Box(
                    low=-1,
                    high=1,
                    shape=(memory * 2,),
                    dtype=self.dtypes["human_pixel"],
                ),
                "depth_image": spaces.Box(
                    low=0,
                    high=1,
                    shape=(memory, depth_width, depth_height),
                    dtype=self.dtypes["depth_image"],
                ),
                # "desired-distance": spaces.Box(low=-0.1, high=4, shape=(1,),  dtype=float),
            }
        )
        # Action space, 2 real values for each motor speed
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # Set render mode and initialize pybullet client
        if render_mode == "DIRECT":
            self._client_id = p.connect(p.DIRECT)
        elif render_mode == "GUI":
            self._client_id = p.connect(p.GUI)

        # Create variable placeholders for pybullet objects
        self._planeId = None
        self._human = None
        self._robot = None
        self._world = None
        self._robot_tracker = None
        self._steps = 0

        self.human_seen = False

    def reset(self, seed=None, options=None):
        self._steps = 0
        super().reset(seed=seed)
        np.random.seed(seed)
        # Reset simulation
        p.resetSimulation(self._client_id)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81, self._client_id)

        # Create plane as floor
        self._planeId = p.loadURDF(
            physicsClientId=self._client_id, fileName="plane.urdf", useFixedBase=True
        )
        # Create world object
        self._world = World(self._client_id)
        # Generate basic room, 4 walls
        self._world.create_basic_room()

        # Get spawning zone for robot and human percentage
        robot_area, human_area = self._world.calculate_starting_areas(area1=1, area2=1)

        # Spawn human
        self._human = Human(self._client_id)
        self._human.reset(starting_area=human_area)

        # Spawn robot
        self._robot = Robot(
            client_id=self._client_id,
            depth_width=self.depth_width,
            depth_height=self.depth_height,
            rgb_width=self.rgb_width,
            rgb_height=self.rgb_height,
        )
        # Check distance between robot and human so they do not collision
        distance = 0
        while distance < 0.4:
            robot_area, _ = self._world.calculate_starting_areas(area1=1, area2=1)
            self._robot.reset(starting_area=robot_area)
            distance = get_human_robot_distance(
                client_id=self._client_id,
                robot_id=self._robot.id,
                human_id=self._human.id,
            )

        self._robot_tracker = RobotTracker(
            client_id=self._client_id,
            robot_id=self._robot.id,
            human_id=self._human.id,
            max_track=self.max_track,
        )

        # Reset observation space and fill it with first values
        self.observation_manager.reset()
        for _ in range(4 * self.memory):
            observation = self._get_observation()
            p.stepSimulation(physicsClientId=self._client_id)
        info = self._get_info()

        self.relevant_collisions = [(self._robot.id, self._human.id)]
        for wall_id in self._world.ids:
            self.relevant_collisions.append((self._robot.id, wall_id))

        self.collision_detector = pyb_utils.CollisionDetector(
            client_id=self._client_id, collision_pairs=self.relevant_collisions
        )
        # Variable to punish robot for losing human in camera
        self.human_seen = False

        return observation, info

    def step(self, action):
        # print(action)
        p.stepSimulation(physicsClientId=self._client_id)
        self._human_walk()
        scaled_action = self._scale_action(action)
        self._move_robot(scaled_action)
        observation = self._get_observation()
        reward = self._compute_reward()
        terminated, truncated = self._get_end_episode()
        if truncated:
            reward = -100
        # This means target achieved by maximum points (at the correct distance and centered on the camera).
        if terminated and reward == 8.0:
            reward = 10000
        info = self._get_info()
        self._steps += 1
        # print("Reward: ", reward)

        return observation, reward, terminated, truncated, info

    def close(self):
        p.disconnect(self._client_id)

    def _human_walk(self):
        # self._human.step()
        pass

    def _move_robot(self, action):
        self._robot.move(action=action)
        self._robot_tracker.track()

    def _compute_reward(self):
        distance = get_human_robot_distance(
            client_id=self._client_id, robot_id=self._robot.id, human_id=self._human.id
        )
        reward_based_on_distance = calculate_distance_reward(
            distance=distance,
            desired_distance=self.desired_distance,
            max_distace=self._world.max_distance,
            offset=self.offset,
        )
        x, any_detected = self.observation_manager.get_x()
        reward_based_ond_pixel = calculate_pixel_reward(
            x=x, any_detected=any_detected, offset=0.1, has_seen=self.human_seen
        )
        reward = reward_based_ond_pixel
        # print(f"Pixel reward: {reward_based_ond_pixel} Distance Reward: {reward_based_on_distance}")
        if reward_based_ond_pixel > 0:
            reward = reward_based_on_distance + reward_based_ond_pixel
        return reward

    def _get_end_episode(self):
        truncated = False
        terminated = False
        if self._steps > self.start_terminated_check:
            terminated = self._robot_tracker.check_proximity(
                desired_distance=self.desired_distance, offset=self.offset
            )
        if self._steps > self.max_steps:
            terminated = True
        if self.collision_detector.in_collision(margin=0.0):
            truncated = True
        return terminated, truncated

    def _get_observation(self):
        rgb, depth = self._robot.get_images()
        x, y = get_human_coordinates(rgb)
        if not self.human_seen and x > -1:
            self.human_seen = True
        speedL, speedR = self._robot.get_motor_speeds()
        observations = {}
        obs = {
            "wheels_speed": np.array([speedL, speedR], dtype=np.float32),
            "human_pixel": np.array([x, y], dtype=np.float32),
            "depth_image": np.array(depth, dtype=np.float32),
        }

        self.observation_manager.add(obs)
        observations = self.observation_manager.get_observations()

        return observations

    def _scale_action(self, norm_action):
        scaled_action = np.array([0, 0])
        scaled_action[0] = (
            norm_action[0] * self.boundings["wheels_speed"][norm_action[0] > 0]
        )
        scaled_action[0] = scaled_action[0] if norm_action[0] > 0 else -scaled_action[0]
        scaled_action[1] = (
            norm_action[1] * self.boundings["wheels_speed"][norm_action[1] > 0]
        )
        scaled_action[1] = scaled_action[1] if norm_action[1] > 0 else -scaled_action[1]
        return scaled_action

    # TODO get current environment information
    def _get_info(self):
        return {}
