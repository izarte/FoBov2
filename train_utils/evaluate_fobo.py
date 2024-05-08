import argparse
from stable_baselines3 import SAC, PPO
import gymnasium as gym
import glob
import os
import fobo2_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

import numpy as np
import torch
torch.cuda.empty_cache()


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
parser.add_argument(
    "-n",
    "--model_name",
    type=str,
    help="Directory containing .zip model files",
)
parser.add_argument(
    "-t",
    "--model_type",
    type=str,
    help="Algorithm used to train model",
)
args = parser.parse_args()


env_kwargs = {
    "render_mode": args.render_mode,  # Use the command line argument
    "memory": 12,
    "rgb_width": 128,
    "rgb_height": 128,
    "depth_width": 320,
    "depth_height": 320,
}

env = make_vec_env(
    "fobo2_env/FoBo2-v0",
    n_envs=1,
    env_kwargs=env_kwargs,
)

print("Observation space:", env.observation_space)
print("Action space:", env.action_space)

if args.model_name:
    # Check if the specified model file exists in the directory
    model_path = os.path.join(args.model_dir, args.model_name)
    if not os.path.isfile(model_path):
        print(f"Model file {args.model_name} not found in the directory.")
        exit(1)
    models = [model_path]
else:
    models = glob.glob(os.path.join(args.model_dir, "*.zip"))
for trained_model in models:
    env.reset()
    print("evaluating", trained_model)
    model_type = args.model_type
    if not model_type:
        if "sac" in trained_model:
             model_type = "sac"
        elif "ppo" in trained_model:
            model_type = "ppo"
        else:
            raise RuntimeError("No model type found in model name")

    with torch.no_grad():
        if model_type == "sac":
            model = SAC.load(trained_model, env=env, device='cpu')
        elif model_type == "ppo":
            model = PPO.load(trained_model, env=env)
        else:
            raise RuntimeError("Bad algorithm name given")

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
