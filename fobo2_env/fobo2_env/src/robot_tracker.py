import pybullet as p
from queue import Queue
import numpy as np

from fobo2_env.src.utils import value_in_range


class RobotTracker:
    def __init__(self, robot_id, human_id, client_id, max_track):
        self.robot_id = robot_id
        self.human_id = human_id
        self.client_id = client_id
        self.tracker = Queue(maxsize=max_track)

    def track(self):
        position, _ = p.getBasePositionAndOrientation(
            physicsClientId=self.client_id, bodyUniqueId=self.robot_id
        )

        if self.tracker.full():
            self.tracker.get()
        self.tracker.put(position[:-1])

    def check_proximity(self, desired_distance, offset):
        human_distance, _ = p.getBasePositionAndOrientation(
            physicsClientId=self.client_id, bodyUniqueId=self.human_id
        )

        points = np.array(self.tracker.queue)
        # Calculate the centroid
        # print("points: ", points)
        centroid = np.mean(points, axis=0)

        distance = np.distance = np.sqrt(
            np.sum((np.array(centroid) - np.array(human_distance[:-1])) ** 2)
        )

        in_range = value_in_range(
            value=distance, center=desired_distance, offset=offset
        )

        # Calculate distances from the centroid
        distances = np.linalg.norm(points - centroid, axis=1)

        # Calculate standard deviation of distances
        std_dev = np.std(distances)
        # print("Standard Deviation of Distances:", std_dev)

        return std_dev < 0.01 or in_range
