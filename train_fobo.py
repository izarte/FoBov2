from stable_baselines3 import A2C, SAC
from stable_baselines3.common.env_util import make_vec_env
import fobo2_env
import os


model_name = "fobo"
try:
    mode = os.environ["TRAIN_MODE"]
except:  # noqa: E722
    mode = "GUI"
try:
    save_path = os.environ["SAVE_MODEL_PATH"]
except:  # noqa: E722
    save_path = ""
# Instantiate the env
env_kwargs = {"render_mode": mode}
vec_env = make_vec_env("fobo2_env/FoBo2-v0", n_envs=1, env_kwargs=env_kwargs)

model = SAC(
    "MultiInputPolicy", vec_env, verbose=1, buffer_size=10000, train_freq=37, seed=37
)
print(model.policy)

model.learn(60000)

model.save(save_path + "/" + model_name)
