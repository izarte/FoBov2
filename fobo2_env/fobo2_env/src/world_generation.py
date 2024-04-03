import pybullet as p
import os
import numpy as np

class World():
    def __init__(self, client_id):
        self.client_id = client_id
        self.ids = []
        self.world_type = 'room'
        self.min_values = (0, 0)
        self.max_values = (0, 0)

    def create_basic_room(self, x : float = 0, y : float = 0):
        self.world_type = 'room'
        wall_orientations = [p.getQuaternionFromEuler([0, 0, 0]), p.getQuaternionFromEuler([0, 0, np.radians(90)])]
        wall_positions = [
            [-5, 0, 0],
            [0, -5, 0],
            [5, 0, 0],
            [0, 5, 0]
        ]

        # Get min values for x y coordinates
        self.min_values = tuple(min(col) for col in zip(*wall_positions))
        # Get max values for for x y coordinates
        self.max_values = tuple(max(col) for col in zip(*wall_positions))

        wall_ids = []
        for i, wall_position in enumerate(wall_positions):
            id = p.loadURDF(
                physicsClientId = self.client_id,
                fileName = os.path.dirname(__file__) + "/models/wall.urdf", 
                basePosition = wall_position,
                baseOrientation = wall_orientations[i % 2],
                useFixedBase = True
            )
            wall_ids.append(id)
        self.ids = wall_ids
    
    def calculate_starting_areas(self, area1 : float = 0.2, area2 = 0.8):
        x_min_value = self.min_values[0] + 0.7 
        y_min_value = self.min_values[1] + 0.7 
        x_max_value = self.max_values[0] - 0.7 
        y_max_value = self.max_values[1] - 0.7 
        x_range = x_max_value - x_min_value
        y_range = y_max_value - y_min_value
        area1_range = (
            (x_min_value, x_min_value + x_range * 0.2),
            (y_min_value, y_min_value + y_range * 0.2),
        )

        area2_range = (
            (x_max_value - x_range * area2, x_max_value),
            (y_max_value - y_range * area2, y_max_value),
        )
        print(area1, area2)

        return area1_range, area2_range






