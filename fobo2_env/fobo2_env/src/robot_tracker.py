import pybullet as p
from queue import Queue
import numpy as np

from fobo2_env.src.utils import value_in_range, orientation_in_grad


class RobotTracker:
    def __init__(self, robot_id, human_id, client_id, max_track):
        self.robot_id = robot_id
        self.human_id = human_id
        self.client_id = client_id
        self.pos_tracker = Queue(maxsize=max_track)
        self.orientation_tracker = Queue(maxsize=max_track)
        self.acc_angle = 0

    def track(self):
        position, robot_orientation = p.getBasePositionAndOrientation(
            physicsClientId=self.client_id, bodyUniqueId=self.robot_id
        )
        # _, _, yaw = orientation_in_grad(
        #     p.getEulerFromQuaternion(quaternion=robot_orientation)
        # )
        _, _, yaw = p.getEulerFromQuaternion(quaternion=robot_orientation)
        yaw = np.pi + yaw if yaw < 0 else yaw
        
        self.track_turns(yaw)
        
        if self.pos_tracker.full():
            self.pos_tracker.get()
        self.pos_tracker.put(position[:-1])

        if self.orientation_tracker.full():
            forgotten_yaw = self.orientation_tracker.get()
            self.track_turns(forgotten_yaw, -1)
        self.orientation_tracker.put(yaw)

    def track_turns(self, yaw, add_method : int = 1):
        if self.orientation_tracker.empty():
            return
        last_yaw = self.orientation_tracker.queue[-1]
        if add_method == -1:
            last_yaw = self.orientation_tracker.queue[0]
        if last_yaw > 3/4 * np.pi and yaw < 1/4 * np.pi:
            incr = np.pi + yaw - last_yaw
        elif last_yaw < 1/4 * np.pi and yaw > 3/4 * np.pi:
            incr = yaw - (np.pi + last_yaw)
        else:
            incr = yaw - last_yaw
        self.acc_angle += incr


    def check_proximity(self, desired_distance, offset):
        human_distance, _ = p.getBasePositionAndOrientation(
            physicsClientId=self.client_id, bodyUniqueId=self.human_id
        )


        points = np.array(self.pos_tracker.queue)
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

        loops = np.abs(self.acc_angle // np.pi) 
        # print("Turns: ", self.acc_angle // np.pi)

        return std_dev < 0.01 or in_range or loops > 3
