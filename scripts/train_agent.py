# Description: Script to train the agent
# air-hockeyを学習するためのスクリプト
# 二つのエージェントを対戦させる

import numpy as np
from air_hockey_challenge.framework import AirHockeyChallengeWrapper
from ppo_air_hockey.agent import PPOAgent

def main(argv=None):
    env = AirHockeyChallengeWrapper(env="7dof-hit", interpolation_order=3)

    agent1 = PPOAgent()

    agents = agent1

    obs = env.reset()
    agents.episode_start()

    steps = 0
    while True:
        steps += 1
        action = agents.draw_action(obs)
        obs, reward, done, info = env.step(action)

        env.render()

        if done or steps > env.info.horizon:
            steps = 0
            obs = env.reset()
            agents.episode_start()
            print("Reset")


if __name__ == '__main__':
    main()