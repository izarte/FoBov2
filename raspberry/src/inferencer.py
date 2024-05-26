import gymnasium as gym
import fobo2_env
from fobo2_env.src.observation_manager import ObservationManager
import numpy as np
from stable_baselines3 import SAC, PPO
import torch
import asyncio
import websockets
import json
import base64

MEMORY = 6
RGB_WIDTH = 1024
MODEL_PATH = "fobo_ppo"


class Inferencer:
    def __init__(self):
        self.env = gym.make("fobo2_env/FoBo2-v0", render_mode="DIRECT")
        self.env.reset()
        self.boundings = {
            "wheels_speed": [-30, 30],
            "human_pixel": [0, RGB_WIDTH],
            "depth_image": [0, 255],
        }
        self.dtypes = {
            "wheels_speed": np.float32,
            "human_pixel": np.float32,
            "depth_image": np.float32,
        }
        self.observation_manager = ObservationManager(
            memory=MEMORY, boundings=self.boundings, dtypes=self.dtypes
        )
        with torch.no_grad():
            # self.model = SAC.load(MODEL_PATH, env=self.env, device="cpu")
            self.model = PPO.load(MODEL_PATH)
        self.init_msgs = 0

    def deserialize_data(self, data_r):
        data = json.loads(data_r)
        bytes_data = base64.b64decode(data["depth_image"])
        data["depth_image"] = np.frombuffer(bytes_data, dtype=float).reshape(1, 8)[0]
        bytes_data = base64.b64decode(data["human_pixel"])
        data["human_pixel"] = np.frombuffer(bytes_data, dtype=float).reshape(1, 2)[0]
        bytes_data = base64.b64decode(data["wheels_speed"])
        data["wheels_speed"] = np.frombuffer(bytes_data, dtype=float).reshape(1, 2)[0]
        return data

    async def listen_and_respond(self, uri="ws://192.168.1.254:8002"):
        async with websockets.connect(uri) as websocket:
            try:
                while True:
                    print("waiting for message")
                    message = await websocket.recv()
                    data = self.deserialize_data(message)
                    print(f"Received message from server: {data}")
                    if self.init_msgs < MEMORY:
                        self.init_msgs += 1
                        continue
                    self.add_observation(data)
                    action = self.inference()
                    await websocket.send(json.dumps({"action": action.tolist()}))
            except websockets.exceptions.ConnectionClosed as e:
                print(f"Server disconnected: {e}")

    def add_observation(self, obs: dict):
        self.observation_manager.add(obs)

    def inference(self):
        obs = self.observation_manager.get_observations()
        with torch.no_grad():
            action, _states = self.model.predict(obs, deterministic=True)
        return action

    def start_listening(self):
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(self.listen_and_respond())
        else:
            loop.run_until_complete(self.listen_and_respond())


if __name__ == "__main__":
    inferencer = Inferencer()
    inferencer.start_listening()
