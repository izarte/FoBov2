from typing import Any, Dict

import numpy as np
import optuna
from stable_baselines3.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)
from torch import nn as nn

from rl_zoo3 import linear_schedule


def sample_custom_env_params(trial: optuna.Trial) -> Dict[str, Any]:
    memory_size = trial.suggest_categorical("memory", [4, 6, 8, 10, 16, 24, 32, 64])
    rgb_size = trial.suggest_categorical("rgb_size", [64, 96, 128, 192, 256, 384, 512])
    depth_size = trial.suggest_categorical(
        "depth_size", [64, 96, 128, 192, 256, 384, 512]
    )
    n_envs = trial.suggest_categorical("n_envs", [2, 4, 8, 16, 32, 64])

    env_kwargs = {
        "memory": memory_size,
        "rgb_width": rgb_size,
        "rgb_height": rgb_size,
        "depth_width": depth_size,
        "depth_height": depth_size,
    }
    return env_kwargs, n_envs


def sample_her_params(
    trial: optuna.Trial, hyperparams: Dict[str, Any], her_kwargs: Dict[str, Any]
) -> list[Dict[str, Any], Dict[str, Any], int]:
    """
    Sampler for HerReplayBuffer hyperparams.

    :param trial:
    :parma hyperparams:
    :return:
    """
    her_kwargs = her_kwargs.copy()
    her_kwargs["n_sampled_goal"] = trial.suggest_int("n_sampled_goal", 1, 5)
    her_kwargs["goal_selection_strategy"] = trial.suggest_categorical(
        "goal_selection_strategy", ["final", "episode", "future"]
    )
    hyperparams["replay_buffer_kwargs"] = her_kwargs
    return hyperparams


def sample_sac_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for SAC hyperparams.

    :param trial:
    :return:
    """
    gamma = trial.suggest_categorical(
        "gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999]
    )
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [256, 384, 512, 896, 1024])
    # buffer_size = trial.suggest_categorical(
    #     "buffer_size", [int(1e4), int(1e5), int(1e6)]
    # )
    buffer_size = 10000
    learning_starts = trial.suggest_categorical(
        "learning_starts", [0, 1000, 10000, 20000]
    )
    # train_freq = trial.suggest_categorical('train_freq', [1, 10, 100, 300])
    train_freq = trial.suggest_categorical(
        "train_freq", [1, 4, 6, 8, 16, 32, 64, 128, 256, 512]
    )
    # Polyak coeff
    tau = trial.suggest_categorical("tau", [0.001, 0.005, 0.01, 0.02, 0.05, 0.08])
    # gradient_steps takes too much time
    # gradient_steps = trial.suggest_categorical('gradient_steps', [1, 100, 300])
    gradient_steps = train_freq
    # ent_coef = trial.suggest_categorical('ent_coef', ['auto', 0.5, 0.1, 0.05, 0.01, 0.0001])
    ent_coef = "auto"
    # You can comment that out when not using gSDE
    log_std_init = trial.suggest_float("log_std_init", -4, 1)
    # NOTE: Add "verybig" to net_arch when tuning HER
    net_arch_type = trial.suggest_categorical("net_arch", ["small", "medium", "big"])
    # activation_fn = trial.suggest_categorical('activation_fn', [nn.Tanh, nn.ReLU, nn.ELU, nn.LeakyReLU])

    net_arch = {
        "small": [64, 64],
        "medium": [256, 256],
        "big": [400, 300],
        # Uncomment for tuning HER
        # "large": [256, 256, 256],
        # "verybig": [512, 512, 512],
    }[net_arch_type]

    target_entropy = "auto"
    # if ent_coef == 'auto':
    #     # target_entropy = trial.suggest_categorical('target_entropy', ['auto', 5, 1, 0, -1, -5, -10, -20, -50])
    #     target_entropy = trial.suggest_float('target_entropy', -10, 10)

    hyperparams = {
        "gamma": gamma,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "learning_starts": learning_starts,
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
        "ent_coef": ent_coef,
        "tau": tau,
        "target_entropy": target_entropy,
        "policy_kwargs": dict(log_std_init=log_std_init, net_arch=net_arch),
    }

    # if additional_args["using_her_replay_buffer"]:
    #     hyperparams = sample_her_params(
    #         trial, hyperparams, additional_args["her_kwargs"]
    #     )
    env_kwargs, n_envs = sample_custom_env_params(trial=trial)

    return hyperparams, env_kwargs, n_envs


def sample_a2c_params(trial: optuna.Trial) -> list[Dict[str, Any], Dict[str, Any], int]:
    """
    Sampler for A2C hyperparams.

    :param trial:
    :return:
    """
    gamma = trial.suggest_categorical(
        "gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999]
    )
    normalize_advantage = trial.suggest_categorical(
        "normalize_advantage", [False, True]
    )
    max_grad_norm = trial.suggest_categorical(
        "max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5]
    )
    # Toggle PyTorch RMS Prop (different from TF one, cf doc)
    use_rms_prop = trial.suggest_categorical("use_rms_prop", [False, True])
    gae_lambda = trial.suggest_categorical(
        "gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0]
    )
    n_steps = trial.suggest_categorical(
        "n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    )
    lr_schedule = trial.suggest_categorical("lr_schedule", ["linear", "constant"])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True)
    vf_coef = trial.suggest_float("vf_coef", 0, 1)
    # Uncomment for gSDE (continuous actions)
    # log_std_init = trial.suggest_float("log_std_init", -4, 1)
    ortho_init = trial.suggest_categorical("ortho_init", [False, True])
    net_arch_type = trial.suggest_categorical("net_arch", ["small", "medium"])
    # sde_net_arch = trial.suggest_categorical("sde_net_arch", [None, "tiny", "small"])
    # full_std = trial.suggest_categorical("full_std", [False, True])
    # activation_fn = trial.suggest_categorical('activation_fn', ['tanh', 'relu', 'elu', 'leaky_relu'])
    activation_fn_name = trial.suggest_categorical("activation_fn", ["tanh", "relu"])

    if lr_schedule == "linear":
        learning_rate = linear_schedule(learning_rate)  # type: ignore[assignment]

    net_arch = {
        "small": dict(pi=[64, 64], vf=[64, 64]),
        "medium": dict(pi=[256, 256], vf=[256, 256]),
    }[net_arch_type]

    # sde_net_arch = {
    #     None: None,
    #     "tiny": [64],
    #     "small": [64, 64],
    # }[sde_net_arch]

    activation_fn = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "elu": nn.ELU,
        "leaky_relu": nn.LeakyReLU,
    }[activation_fn_name]

    hyperparams = {
        "n_steps": n_steps,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "normalize_advantage": normalize_advantage,
        "max_grad_norm": max_grad_norm,
        "use_rms_prop": use_rms_prop,
        "vf_coef": vf_coef,
        "policy_kwargs": dict(
            # log_std_init=log_std_init,
            net_arch=net_arch,
            # full_std=full_std,
            activation_fn=activation_fn,
            # sde_net_arch=sde_net_arch,
            ortho_init=ortho_init,
        ),
    }
    env_kwargs, n_envs = sample_custom_env_params(trial=trial)

    return hyperparams, env_kwargs, n_envs


def sample_ppo_params(trial: optuna.Trial) -> list[Dict[str, Any], Dict[str, Any], int]:
    """
    Sampler for PPO hyperparams.

    :param trial:
    :return:
    """
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512])
    n_steps = trial.suggest_categorical(
        "n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    )
    gamma = trial.suggest_categorical(
        "gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999]
    )
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True)
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
    n_epochs = trial.suggest_categorical("n_epochs", [1, 5, 10, 20])
    gae_lambda = trial.suggest_categorical(
        "gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0]
    )
    max_grad_norm = trial.suggest_categorical(
        "max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5]
    )
    vf_coef = trial.suggest_float("vf_coef", 0, 1)
    net_arch_type = trial.suggest_categorical("net_arch", ["tiny", "small", "medium"])
    # Uncomment for gSDE (continuous actions)
    # log_std_init = trial.suggest_float("log_std_init", -4, 1)
    # Uncomment for gSDE (continuous action)
    # sde_sample_freq = trial.suggest_categorical("sde_sample_freq", [-1, 8, 16, 32, 64, 128, 256])
    # Orthogonal initialization
    ortho_init = False
    # ortho_init = trial.suggest_categorical('ortho_init', [False, True])
    # activation_fn = trial.suggest_categorical('activation_fn', ['tanh', 'relu', 'elu', 'leaky_relu'])
    activation_fn_name = trial.suggest_categorical("activation_fn", ["tanh", "relu"])
    # lr_schedule = "constant"
    # Uncomment to enable learning rate schedule
    # lr_schedule = trial.suggest_categorical('lr_schedule', ['linear', 'constant'])
    # if lr_schedule == "linear":
    #     learning_rate = linear_schedule(learning_rate)

    # TODO: account when using multiple envs
    if batch_size > n_steps:
        batch_size = n_steps

    # Independent networks usually work best
    # when not working with images
    net_arch = {
        "tiny": dict(pi=[64], vf=[64]),
        "small": dict(pi=[64, 64], vf=[64, 64]),
        "medium": dict(pi=[256, 256], vf=[256, 256]),
    }[net_arch_type]

    activation_fn = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "elu": nn.ELU,
        "leaky_relu": nn.LeakyReLU,
    }[activation_fn_name]

    env_kwargs, n_envs = sample_custom_env_params(trial=trial)

    kwargs = {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "clip_range": clip_range,
        "n_epochs": n_epochs,
        "gae_lambda": gae_lambda,
        "max_grad_norm": max_grad_norm,
        "vf_coef": vf_coef,
        # "sde_sample_freq": sde_sample_freq,
        "policy_kwargs": dict(
            # log_std_init=log_std_init,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
        ),
    }

    return kwargs, env_kwargs, n_envs


def sample_ppo_lstm_params(
    trial: optuna.Trial, n_actions: int, n_envs: int, additional_args: dict
) -> list[Dict[str, Any], Dict[str, Any], int]:
    """
    Sampler for RecurrentPPO hyperparams.
    uses sample_ppo_params(), this function samples for the policy_kwargs
    :param trial:
    :return:
    """
    hyperparams, env_kwargs, n_envs = sample_ppo_params(
        trial, n_actions, n_envs, additional_args
    )

    enable_critic_lstm = trial.suggest_categorical("enable_critic_lstm", [False, True])
    lstm_hidden_size = trial.suggest_categorical(
        "lstm_hidden_size", [16, 32, 64, 128, 256, 512]
    )

    hyperparams["policy_kwargs"].update(
        {
            "enable_critic_lstm": enable_critic_lstm,
            "lstm_hidden_size": lstm_hidden_size,
        }
    )

    return hyperparams, env_kwargs, n_envs
