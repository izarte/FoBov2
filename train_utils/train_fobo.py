from stable_baselines3 import SAC, A2C, PPO
from stable_baselines3.common.env_util import make_vec_env
import torch
import fobo2_env
import os
import json
from pathlib import Path

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv


ENV_ID = "fobo2_env/FoBo2-v0"
MODEL_NAME = "fobo"


def main():
    try:
        gpu = int(os.environ["USED_GPU"])
    except:  # noqa: E722
        gpu = 0
    try:
        mode = os.environ["TRAIN_MODE"]
    except:  # noqa: E722
        # mode = "DIRECT"
        mode = "GUI"
    try:
        save_path = Path(os.environ["SAVE_MODEL_PATH"])
    except:  # noqa: E722
        save_path = ""
    try:
        model_type = os.environ["MODEL_TYPE"]
    except:  # noqa: E722
        model_type = "sac"
    save_path = f"{save_path / MODEL_NAME}_{model_type}_0"
    counter = 1
    while os.path.exists(save_path):
        # If the folder exists, append a number to the base name and increment the counter
        save_path = save_path[:-1] + str(counter)
        counter += 1
    # Create the folder with the unique name
    os.makedirs(save_path)
    os.makedirs(save_path + "/logs")
    os.makedirs(save_path + "/checkpoints")

    torch.cuda.set_device(gpu)
    device = torch.cuda.current_device() if torch.cuda.is_available() else "CPU"
    print("PyTorch is using device:", device)
    train(mode, Path(save_path), model_type)


def train(mode, save_path, model_type):
    # Instantiate the env
    # Save a checkpoint every 1000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path=save_path / "checkpoints",
        name_prefix=MODEL_NAME,
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    env_kwargs = {"render_mode": mode, "memory": 4}
    log_dir = save_path / "logs"

    model_name = MODEL_NAME + "_" + model_type
    if model_type == "sac":
        with open("hyperparameters/sac_hyperparameters.json", "r") as file:
            data = json.load(file)
        # # n_envs = data["Best_trial"]["n_envs"]
        n_envs = 6
        env_kwargs.update(data["Best_trial"]["env"])
        print(env_kwargs)
        vec_env = make_vec_env(
            "fobo2_env/FoBo2-v0",
            n_envs=n_envs,
            monitor_dir=log_dir,
            # monitor_kwargs=monitor_kwargs,
            env_kwargs=env_kwargs,
            vec_env_cls=SubprocVecEnv,
        )
        kwargs = {"policy": "MultiInputPolicy", "env": vec_env}
        # kwargs.update(data["Best_trial"]["Params"])
        c_kwargs = {
            "buffer_size": 10000,
            "ent_coef": "auto",
            "train_freq": 4,
            "seed": 37,
            "batch_size": 256,
        }
        kwargs.update(c_kwargs)

        # model = SAC(
        #     "MultiInputPolicy",
        #     vec_env,
        #     verbose=1,
        #     buffer_size=10000,
        #     ent_coef="auto",
        #     train_freq=3,
        #     seed=37,
        #     batch_size=256,
        # )
        model = SAC(**kwargs)

    elif model_type == "a2c":
        model = A2C(
            "MultiInputPolicy",
            vec_env,
            verbose=1,
            ent_coef=0.01,
            n_steps=5,
            gamma=0.99,
            seed=37,
        )

        print(model.policy)

    filtered_kwargs = {key: value for key, value in kwargs.items() if key != "env"}
    hyperparameters = {
        "model": filtered_kwargs,
        "environment": env_kwargs,
    }
    print(hyperparameters)
    hyperparameters_path = save_path / "hyperparametrs.json"
    with open(hyperparameters_path, "w") as f:
        json.dump(hyperparameters, f, indent=4)
    print(model.policy)
    # model.learn(5, callback=checkpoint_callback)
    model.learn(n_envs * 500000, callback=checkpoint_callback)
    model_save_path = save_path / model_name
    print("Model succesfully learnt, saving in ", model_save_path)
    model.save(model_save_path)


if __name__ == "__main__":
    main()
