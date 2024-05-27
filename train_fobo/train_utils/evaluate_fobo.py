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
import json

torch.cuda.empty_cache()


def get_metrics(args):
    checkpoints = []

    # Check all files in the specified directory
    for file in os.listdir(args.model_dir):
        # Check if the file is a zip file
        if file.endswith(".zip"):
            # Add it to the list
            checkpoints.append(args.model_dir + "/" + os.path.splitext(file)[0])
    model_type = args.model_type
    if not model_type:
        if "sac" in args.model_dir:
            env_kwargs = {
                "render_mode": "DIRECT",  # Use the command line argument
                "memory": 4,
                "rgb_width": 96,
                "rgb_height": 96,
                "depth_width": 64,
                "depth_height": 64,
            }
            model_type = "sac"
        elif "ppo" in args.model_dir:
            env_kwargs = {
                "render_mode": "DIRECT",  # Use the command line argument
                "memory": 32,
                "rgb_width": 192,
                "rgb_height": 192,
                "depth_width": 64,
                "depth_height": 64,
            }
            model_type = "ppo"
        else:
            raise RuntimeError("No model type found in model name")

    env = make_vec_env(
        "fobo2_env/FoBo2-v0",
        n_envs=1,
        env_kwargs=env_kwargs,
    )

    results_file_path = f"{args.model_dir}/results_{model_type}.json"

    # Initial check to create the file or read existing data
    try:
        with open(results_file_path, "r") as file:
            results = json.load(file)
    except FileNotFoundError:
        results = []

    for checkpoint in checkpoints:
        with torch.no_grad():
            if model_type == "sac":
                model = SAC.load(checkpoint, env=env)
            elif model_type == "ppo":
                model = PPO.load(checkpoint, env=env)
            else:
                raise RuntimeError("Bad algorithm name given")
            reward_list, std_reward = evaluate_policy(
                model, model.get_env(), n_eval_episodes=10, return_episode_rewards=True
            )

        print(reward_list, std_reward)

        # Append results to the list
        results.append(
            {
                "Checkpoint": checkpoint.split("_")[-2],
                "Model Type": model_type,
                "Reward_list": reward_list,
                "Steps": std_reward,
            }
        )

        # Save the updated results list to a JSON file after each iteration
        with open(results_file_path, "w") as json_file:
            json.dump(results, json_file, indent=4)


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

parser.add_argument(
    "-gm",
    "--get_metrics",
    action="store_true",
    default=False,
    help="Get metrics for each checkpoint",
)

args = parser.parse_args()


if args.get_metrics:
    get_metrics(args)
else:
    # env_kwargs = {
    #     "render_mode": args.render_mode,  # Use the command line argument
    #     "memory": 6,
    #     "rgb_width": 128,
    #     "rgb_height": 128,
    #     "depth_width": 320,
    #     "depth_height": 320,
    # }

    env_kwargs = {
        "render_mode": args.render_mode,  # Use the command line argument
        "memory": 4,
        "rgb_width": 96,
        "rgb_height": 96,
        "depth_width": 64,
        "depth_height": 64,
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
            print(f"Model file {args.model_name} not found in the {model_path}.")
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
                model = SAC.load(trained_model, env=env, device="cpu")
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
