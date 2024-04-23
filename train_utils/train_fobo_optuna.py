from hyperparameters_samples import sample_a2c_params, sample_sac_params

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
    # new_kwargs, new_env_kwargs, n_envs = sample_a2c_params(trial)
    new_kwargs, new_env_kwargs, n_envs = sample_sac_params(trial)
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

    eval_env = Monitor(vec_env)

    eval_callback = TrialEvalCallback(
        eval_env,
        trial,
        n_eval_episodes=N_EVAL_EPISODES,
        eval_freq=EVAL_FREQ,
        deterministic=True,
    )

    kwargs["env"] = vec_env
    # Create the RL model.
    oversize = False
    try:
        # model = A2C(**kwargs)
        model = SAC(**kwargs)
    except Exception as e:
        print(e)
        oversize = True
    if oversize:
        return float("nan")
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
