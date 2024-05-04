import numpy as np
import gymnasium as gym
import fobo2_env
from stable_baselines3 import PPO

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import numpy as np
import matplotlib.pyplot as plt

def calculate_exponential_score(max_score, value, desired_value, k, is_distance = False) -> float:
    delta = np.abs(value - desired_value)
    A = max_score

    if is_distance and value < desired_value:
        delta = value
        A = -3
        
    reward = A * np.exp(-k * delta)

    return reward


def plot_reward_function():
    x_values = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 9, 10, 11, 12,13, 14]
    y_values = []
    for i in [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 9, 10, 11, 12,13, 14]:
        y = calculate_exponential_score(2, i, 1.5, 0.2, True)
        y_values.append(y)
        print(f"for {i} distance, reward = {y}")

    # Add a vertical line at x = 1.5, desired distance
    plt.axvline(x=1.5, color='r', linestyle='--', label='x = 1.5')  # Adjust color and linestyle as needed

    # Plot the function
    plt.plot(x_values, y_values, label='sin(x)')  # Change the label as needed
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('reward')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_reward_function_px():
    x_values = [i / 20 for i in range(0, 21)]
    y_values = []
    for i in x_values:
        y = calculate_exponential_score(2, i, 0.5, 6)
        y_values.append(y)
        print(f"for {i} distance, reward = {y}")

    # Add a vertical line at x = 1.5, desired distance
    plt.axvline(x=0.5, color='r', linestyle='--', label='x = 0.5')  # Adjust color and linestyle as needed

    # Plot the function
    plt.plot(x_values, y_values, label='sin(x)')  # Change the label as needed
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('reward')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    plot_reward_function()
    plot_reward_function_px()
