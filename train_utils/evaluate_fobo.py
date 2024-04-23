import argparse
from stable_baselines3 import SAC
import gymnasium as gym
import glob
import os
import fobo2_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

import numpy as np

# Set up argparse
parser = argparse.ArgumentParser(
    description="Run FoBo2 environment with different render modes."
)
parser.add_argument(
    "-r",
    "--render_mode",
    choices=["GUI", "DIRECT"],
    default="GUI",
    help="Select the render mode: GUI or DIRECT",
)
parser.add_argument(
    "-d",
    "--model_dir",
    type=str,
    default="trained_models",
    help="Directory containing .zip model files",
)
args = parser.parse_args()


env_kwargs = {
    "render_mode": args.render_mode,  # Use the command line argument
    "memory": 4,
    "depth_width": 128,
    "depth_height": 128,
}

env = make_vec_env(
    "fobo2_env/FoBo2-v0",
    n_envs=1,
    env_kwargs=env_kwargs,
)

print("Observation space:", env.observation_space)
print("Action space:", env.action_space)

zip_files = glob.glob(os.path.join(args.model_dir, "*.zip"))
for trained_model in zip_files:
    env.reset()
    print("evaluating", trained_model)
    model = SAC.load(trained_model, env=env)
    mean_reward, std_reward = evaluate_policy(
        model, model.get_env(), n_eval_episodes=10
    )
    print("Mean reward: ", mean_reward)
    print("Standard reward", std_reward)


# # Enjoy trained agent
# vec_env = model.get_env()
# obs = env.reset()
# for i in range(1000):
#     # action, _states = model.predict(obs, deterministic=True)
#     observation, reward, terminated, truncated, info = env.step([1, 1])
