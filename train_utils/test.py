import numpy as np
import gymnasium as gym
import fobo2_env
from stable_baselines3 import PPO

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv


def calculate_exponential_score(max_score, value, desired_value) -> float:
    delta = np.abs(value - desired_value)
    A = max_score
    k = 1
    reward = A * np.exp(-k * delta)

    return reward


def main():
    env_kwargs = {
        "render_mode": "GUI",  # Use the command line argument
        "memory": 2,
        "rgb_width": 18,
        "rgb_height": 18,
        "depth_width": 18,
        "depth_height": 18,
    }

    print(calculate_exponential_score(1, 18, 1.5))
    env = make_vec_env(
        "fobo2_env/FoBo2-v0",
        n_envs=1,
        env_kwargs=env_kwargs,
        vec_env_cls=SubprocVecEnv,
    )
    model = PPO(policy="MultiInputPolicy", env=env)

    # env = gym.make("fobo2_env/FoBo2-v0")
    # vec_env = model.get_env()
    obs = env.reset()
    print(env.action_space)
    for i in range(1000):
        # action, _states = model.predict(obs, deterministic=True)
        a = model.predict(obs)
        print("Action: ", a)
        obs, reward, terminated, truncated = env.step(a[0])


if __name__ == "__main__":
    main()
