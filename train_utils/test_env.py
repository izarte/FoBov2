import gymnasium as gym
import fobo2_env
from stable_baselines3.common.env_checker import check_env

env = gym.make("fobo2_env/FoBo2-v0")
# It will check your custom environment and output additional warnings if needed
check_env(env)
