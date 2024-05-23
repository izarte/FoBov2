import os

import numpy as np
import pybullet as p

from fobo2_env.src.utils import random_pos_orientation

class World:
    def __init__(self, client_id):
        self.client_id = client_id
        self.ids = []
        self.world_type = "room"
        self.min_values = (0, 0)
        self.max_values = (0, 0)
        self.max_distance = 0
        self.obstacles_types = ["box", "wall_like", "sq_column"]
        self.x_value_obst = {"box": 0.5, "wall_like": 0.5, "sq_column": 0.5}
        self.y_value_obst = {"box": 0.5, "wall_like": 1.5, "sq_column": 0.5}
        self.z_value_obst = {"box": 0.5, "wall_like": 0.5, "sq_column": 1.5}
        self.obstacles_list = []
        self.obstacles_corners = []

    def create_basic_room(self, x: float = 0, y: float = 0):
        self.world_type = "room"
        wall_orientations = [
            p.getQuaternionFromEuler([0, 0, 0]),
            p.getQuaternionFromEuler([0, 0, np.radians(90)]),
        ]
        wall_length = 15
        wall_positions = [
            [-wall_length, 0, 0],
            [0, -wall_length, 0],
            [wall_length, 0, 0],
            [0, wall_length, 0],
        ]
        self.max_distance = np.sqrt(2 * wall_length**2)
        # Get min values for x y coordinates
        self.min_values = tuple(min(col) for col in zip(*wall_positions))
        # Get max values for for x y coordinates
        self.max_values = tuple(max(col) for col in zip(*wall_positions))

        wall_ids = []
        for i, wall_position in enumerate(wall_positions):
            id = p.loadURDF(
                physicsClientId=self.client_id,
                fileName=os.path.dirname(__file__) + "/models/wall.urdf",
                basePosition=wall_position,
                baseOrientation=wall_orientations[i % 2],
                useFixedBase=True,
            )
            wall_ids.append(id)
        self.ids = wall_ids
        self.spawn_obstacles()


    def spawn_obstacles(self):
        n_objects = np.random.randint(2, 10)
        self.obstacles_list = []
        self.obstacles_corners = []
        area = self.get_area()
        for _ in range(n_objects):
            obstacle_type = np.random.choice(self.obstacles_types)
            area = (area[0], area[1], self.z_value_obst[obstacle_type])
            random_pos, random_orientation = random_pos_orientation(area)
            _, _, yaw = p.getEulerFromQuaternion(random_orientation)
            corners = self.generate_corners_coords(random_pos, yaw, self.x_value_obst[obstacle_type], self.y_value_obst[obstacle_type])
            self.obstacles_corners.append(corners)
            obst_id = p.loadURDF(
                physicsClientId=self.client_id,
                fileName=os.path.dirname(__file__) + f"/models/{obstacle_type}.urdf",
                basePosition=random_pos,
                baseOrientation=random_orientation,
                useFixedBase=True,
            )
            self.obstacles_list.append(obst_id)


    def generate_corners_coords(self, origin, yaw, x_length, y_length):

        # Calculate half-diagonals
        diagonal = np.sqrt((x_length) ** 2 + (y_length) ** 2)
        alpha = np.arctan2(x_length, y_length)

        x_length *= 2
        y_length *= 2
        # Calculate corners
        corners = []
        for i in range(4):
            angle = yaw + alpha + i * np.pi / 2
            x = origin[0] + diagonal * np.cos(angle)
            y = origin[1] + diagonal * np.sin(angle)
            corners.append((x, y, origin[2]))

        return corners

    def calculate_starting_areas(self, area1: float = 0.2, area2=0.8):
        x_min_value = self.min_values[0] + 0.7
        y_min_value = self.min_values[1] + 0.7
        x_max_value = self.max_values[0] - 0.7
        y_max_value = self.max_values[1] - 0.7
        x_range = x_max_value - x_min_value
        y_range = y_max_value - y_min_value
        area1_range = (
            (x_min_value, x_min_value + x_range * area1),
            (y_min_value, y_min_value + y_range * area1),
        )

        area2_range = (
            (x_max_value - x_range * area2, x_max_value),
            (y_max_value - y_range * area2, y_max_value),
        )

        return area1_range, area2_range
    
    def get_area(self):
        area = (
            (self.min_values[0] + 0.7, self.max_values[0] - 0.7),
            (self.min_values[1] + 0.7, self.max_values[1] - 0.7),
        )

        return area
