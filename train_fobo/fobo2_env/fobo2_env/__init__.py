from gymnasium.envs.registration import register

register(
     id="fobo2_env/FoBo2-v0",
     entry_point="fobo2_env.envs:FoBo2Env"
)