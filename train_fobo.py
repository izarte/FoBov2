from typing import Any
from typing import Dict

from stable_baselines3 import SAC, A2C, PPO
from stable_baselines3.common.env_util import make_vec_env
import torch
import fobo2_env
import os


from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv


ENV_ID = "fobo2_env/FoBo2-v0"
MODEL_NAME = "fobo_a2c"


def train(gpu, mode, save_path):
    torch.cuda.set_device(gpu)
    # Instantiate the env
    # Save a checkpoint every 1000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path=save_path + "/logs/",
        name_prefix=MODEL_NAME,
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    env_kwargs = {"render_mode": mode, "memory": 10}
    log_dir = save_path + "/logs/"

    vec_env = make_vec_env(
        "fobo2_env/FoBo2-v0",
        n_envs=10,
        monitor_dir=log_dir,
        # monitor_kwargs=monitor_kwargs,
        env_kwargs=env_kwargs,
        vec_env_cls=SubprocVecEnv,
    )

    # model = SAC(
    #     "MultiInputPolicy",
    #     vec_env,
    #     verbose=10,
    #     buffer_size=10000,
    #     ent_coef="auto",
    #     train_freq=3,
    #     seed=37,
    #     batch_size=256,
    # )

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

    # print(model.policy_loss, model.entropy_loss, model.value_loss)
    model.learn(400000, callback=checkpoint_callback)

    model.save(save_path + "/" + MODEL_NAME)


if __name__ == "__main__":
    try:
        gpu = int(os.environ["USED_GPU"])
    except:  # noqa: E722
        gpu = 0
    try:
        mode = os.environ["TRAIN_MODE"]
    except:  # noqa: E722
        mode = "DIRECT"
        # mode = "GUI"
    try:
        save_path = os.environ["SAVE_MODEL_PATH"]
    except:  # noqa: E722
        save_path = ""
    train(gpu, mode, save_path)
    # train_optuna(gpu, mode, save_path)
