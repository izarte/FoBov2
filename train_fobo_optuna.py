from typing import Any
from typing import Dict

from stable_baselines3 import SAC, A2C, PPO
from stable_baselines3.common.env_util import make_vec_env
import torch
import torch.nn as nn
import fobo2_env
import os
import gymnasium as gym

import logging

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from rl_zoo3 import linear_schedule

N_TRIALS = 100
N_STARTUP_TRIALS = 5
N_EVALUATIONS = 2
N_TIMESTEPS = int(2e4)
EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)
N_EVAL_EPISODES = 3

ENV_ID = "fobo2_env/FoBo2-v0"

DEFAULT_HYPERPARAMS = {
    "policy": "MultiInputPolicy",
    "env": ENV_ID,
}


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
    batch_size = trial.suggest_categorical(
        "batch_size", [16, 32, 64, 128, 256, 512, 1024, 2048]
    )
    buffer_size = trial.suggest_categorical(
        "buffer_size", [int(1e4), int(1e5), int(1e6)]
    )
    learning_starts = trial.suggest_categorical(
        "learning_starts", [0, 1000, 10000, 20000]
    )
    # train_freq = trial.suggest_categorical('train_freq', [1, 10, 100, 300])
    train_freq = trial.suggest_categorical(
        "train_freq", [1, 4, 8, 16, 32, 64, 128, 256, 512]
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

    memory_size = trial.suggest_categorical("memory", [4, 6, 8, 10, 16, 24, 32, 64])
    rgb_size = trial.suggest_categorical("rgb_size", [16, 32, 64, 128, 256, 512])
    depth_size = trial.suggest_categorical("depth_size", [16, 32, 64, 128, 256, 512])

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

    env_kwargs = {
        "memory": memory_size,
        "rgb_width": rgb_size,
        "rgb_height": rgb_size,
        "depth_width": depth_size,
        "depth_height": depth_size,
    }

    return hyperparams, env_kwargs


def sample_a2c_params(trial: optuna.Trial) -> Dict[str, Any]:
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

    memory_size = trial.suggest_categorical("memory", [4, 6, 8, 10, 16, 24, 32, 64])
    rgb_size = trial.suggest_categorical(
        "rgb_size", [32, 64, 96, 128, 192, 256, 384, 512]
    )
    depth_size = trial.suggest_categorical(
        "depth_size", [32, 64, 96, 128, 192, 256, 384, 512]
    )
    n_envs = trial.suggest_categorical("n_envs", [2, 4, 8, 16, 32, 64])

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
    env_kwargs = {
        "memory": memory_size,
        "rgb_width": rgb_size,
        "rgb_height": rgb_size,
        "depth_width": depth_size,
        "depth_height": depth_size,
    }

    return hyperparams, env_kwargs, n_envs


class TrialEvalCallback(EvalCallback):
    """Callback used for evaluating and reporting a trial."""

    def __init__(
        self,
        eval_env: gym.Env,
        trial: optuna.Trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        deterministic: bool = True,
        verbose: int = 0,
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune trial if need.
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True


def objective(trial: optuna.Trial) -> float:
    kwargs = DEFAULT_HYPERPARAMS.copy()
    # Sample hyperparameters.
    new_kwargs, new_env_kwargs, n_envs = sample_a2c_params(trial)
    kwargs.update(new_kwargs)
    env_kwargs = {"render_mode": "DIRECT"}
    env_kwargs.update(new_env_kwargs)
    log_dir = save_path + "/logs/"
    vec_env = make_vec_env(
        DEFAULT_HYPERPARAMS["env"],
        n_envs=n_envs,
        monitor_dir=log_dir,
        env_kwargs=env_kwargs,
        vec_env_cls=SubprocVecEnv,
    )

    eval_env = Monitor(gym.make(ENV_ID))

    eval_callback = TrialEvalCallback(
        eval_env,
        trial,
        n_eval_episodes=N_EVAL_EPISODES,
        eval_freq=EVAL_FREQ,
        deterministic=True,
    )

    kwargs["env"] = vec_env
    # Create the RL model.
    model = A2C(**kwargs)
    print(model.policy)
    # Create env used for evaluation.
    # Create the callback that will periodically evaluate and report the performance.
    nan_encountered = False
    try:
        model.learn(N_TIMESTEPS, callback=eval_callback)
    except Exception as e:
        # Sometimes, random hyperparams can generate NaN.
        print(e)
        nan_encountered = True
    finally:
        # Free memory.
        model.env.close()
        eval_env.close()

    # Tell the optimizer that the trial failed.
    if nan_encountered:
        return float("nan")

    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()

    return eval_callback.last_mean_reward


def train_optuna(gpu, mode, save_path):
    log_file = save_path + "/logs/optuna_log.txt"
    # Create the log file if it doesn't exist
    if not os.path.exists(log_file):
        with open(log_file, "w"):
            pass
    logging.basicConfig(filename=log_file, level=logging.INFO)
    torch.cuda.set_device(gpu)
    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    # Do not prune before 1/3 of the max budget is used.
    pruner = MedianPruner(
        n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS // 3
    )

    study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")
    try:
        study.optimize(objective, n_trials=N_TRIALS, timeout=600)
    except KeyboardInterrupt:
        pass

    # Write to log file
    logging.info("Number of finished trials: %d", len(study.trials))

    logging.info("Best trial:")
    trial = study.best_trial

    logging.info("  Value: %f", trial.value)

    logging.info("  Params: ")
    for key, value in trial.params.items():
        logging.info("    %s: %s", key, value)

    logging.info("  User attrs:")
    for key, value in trial.user_attrs.items():
        logging.info("    %s: %s", key, value)


if __name__ == "__main__":
    try:
        gpu = int(os.environ["USED_GPU"])
    except:  # noqa: E722
        gpu = 0

    model_name = "fobo"
    try:
        mode = os.environ["TRAIN_MODE"]
    except:  # noqa: E722
        mode = "DIRECT"
        # mode = "GUI"
    try:
        save_path = os.environ["SAVE_MODEL_PATH"]
    except:  # noqa: E722
        save_path = ""
    train_optuna(gpu, mode, save_path)
