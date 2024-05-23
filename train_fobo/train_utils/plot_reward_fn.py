import numpy as np
import gymnasium as gym
import fobo2_env
from stable_baselines3 import PPO

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import numpy as np
import matplotlib.pyplot as plt


def value_in_range(value: float, center: float, offset: float):
    difference = np.abs(value - center)
    if difference <= offset:
        return True
    return False


def calculate_exponential_score(max_score, value, desired_value, k) -> float:
    delta = np.abs(value - desired_value)
    A = max_score

    # if  value < desired_value:
    #     delta = value
    #     A = -3

    reward = A * np.exp(-k * delta)

    return reward


def plot_reward_function():
    # x_values = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 9, 10, 11, 12,13, 14]
    x_values = [-1] + [i / 1000 for i in range(0, 1000)]
    y_values = []
    for i in x_values:
        # y = calculate_exponential_score(max_score=3, value=i, desired_value=1.5, k=0.2)
        y = calculate_exponential_score(max_score=2, value=i, desired_value=0.5, k=6)

        if value_in_range(
            value=i,
            center=0.5,
            offset=0.1,
        ):
            y = 3

        if i == -1:
            y = 0

        y_values.append(y)
        print(f"for {i} distance, reward = {y}")

    # Add a vertical line at x = 1.5, desired distance
    plt.axvline(
        x=0.5, color="r", linestyle="--", label="x = 0.5"
    )  # Adjust color and linestyle as needed

    # Plot the function
    plt.plot(x_values, y_values, label="r2(x)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("reward based on pixel")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    plot_reward_function()
    # plot_reward_function_px()
