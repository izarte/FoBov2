import gymnasium as gym
import fobo2_env
import cv2


env = gym.make("fobo2_env/FoBo2-v0")
env.reset()

# obs = env.reset()
for i in range(1000):
    observation, reward, terminated, truncated, info = env.step([1, 1])
    depth_image = 255 * observation["depth_image"][0]
    print(depth_image)
    cv2.imshow("depth", depth_image)
    cv2.waitKey(1)
