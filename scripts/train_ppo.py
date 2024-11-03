import sys
print("Python executable path:", sys.executable)
print("Python version:", sys.version)

from stable_baselines3 import PPO
from stable_baselines3.common.envs import DummyVecEnv
from air_hockey_challenge.framework import AirHockeyChallengeWrapper

env = AirHockeyChallengeWrapper(env="7dof-hit", interpolation_order=3)
env = DummyVecEnv([lambda: env])

model = PPO("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=100)

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones:
        obs = env.reset()