from queue import Queue
import numpy as np


class ObservationManager:
    def __init__(self, memory: int, boundings: dict, dtypes: dict):
        self.boundings = boundings
        self.dtypes = dtypes

        self.observations = {
            "wheels_speed": Queue(maxsize=memory),
            "human_pixel": Queue(maxsize=memory),
            "depth_image": Queue(maxsize=memory),
        }

    def reset(self):
        for key in self.observations:
            # Clearing the queue
            while not self.observations[key].empty():
                self.observations[key].get()

    def add(self, obs: dict):
        for key in obs:
            if self.observations[key].full():
                self.observations[key].get()
            obs[key] = self.normalize(obs[key], key)
            self.observations[key].put(obs[key])

    def get_observations(self) -> dict:
        obs = {}
        for key in self.observations:
            obs[key] = np.array(self.observations[key].queue, dtype=self.dtypes[key])
            if key == "wheels_speed" or key == "human_pixel":
                obs[key] = obs[key].flatten()
        return obs

    def normalize(self, values: list, key: str):
        if key == "human_pixel" and values[0] == -1 and values[1] == -1:
            return values
        values = (values - self.boundings[key][0]) / (
            self.boundings[key][1] - self.boundings[key][0]
        )
        return values

    def get_x(self):
        any_detected = False
        for x, _ in self.observations["human_pixel"].queue:
            if x > -1:
                any_detected = True
                break

        # Get last pixel observation specially x
        return self.observations["human_pixel"].queue[-1][0], any_detected
