# Description: Script to train the agent
# air-hockeyを学習するためのスクリプト
# 二つのエージェントを対戦させる

import numpy as np
from air_hockey_challenge.framework import AirHockeyChallengeWrapper
from ppo_air_hockey.agent import PPOAgent

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO

import gym
from gym import spaces
import gymnasium
from stable_baselines3.common.env_checker import check_env

from air_hockey_challenge.framework.agent_base import AgentBase
from baseline.baseline_agent.tactics import *
from air_hockey_challenge.utils.kinematics import forward_kinematics, inverse_kinematics, jacobian


# class PPOAgent(AgentBase):
#     def __init__(self, env_info, agent_id, **kwargs):
#         self.env_info = env_info
#         self.agent_id = agent_id

#         env = DummyVecEnv([lambda: AirHockeyChallengeWrapper(env=self.env_info['env_name'], interpolation_order=3)])
#         self.model = PPO("MlpPolicy", env, verbose=1)

#     @property
#     def observation_space(self):
#         """observation"""
#         return spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32)

#     @property
#     def action_space(self):
#         """action"""
#         return spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

#     def draw_action(self,obs: np.ndarray)->np.ndarray:
#         """obsを与えて行動を返す"""
#         return self.infer_action(self.process_raw_obs(obs))
    
#     def infer_action(self,obs: np.ndarray)->np.ndarray:
#         return self.process_raw_act()
    
#     def initialize_inference(self):

#     def reset(self):

#     def process_raw_obs(self,obs: np.ndarray)->np.ndarray:
#         """obsの正規化による前処理"""
#         return obs
    
#     def process_raw_act(self,act: np.ndarray)->np.ndarray:
#         """actの前処理（ターゲット位置から関節角まで）"""
#         return act
    
#     @classmethod
#     def _normalize_value(cls,value: np.ndarray,low_in: float,high_in: float)->np.ndarray:
#         """値を正規化"""
#         return (2*(value-low_in))/(high_in-low_in) - 1
    
#     def _unnormalize_value(cls,value: np.ndarray,low_out: float,high_out: float)->np.ndarray:
#         """値を非正規化"""
#         return (value+1)*(high_out-low_out)/2 + low_out


#     ###以下はUtil関数###

#     def extract_env_info(self):
#         # table
#         # puck
#         # mullet   
#         # robot
#         # simulation
#         # task

#     def forward_kinematics(self, q, link='ee'):
#         return forward_kinematics(self.robot_model, self.robot_data, q, link)
    
#     def inverse_kinematics(self, desired_position, desired_rotation=None, q_init=None, link='ee'):
#         return inverse_kinematics(self.robot_model, self.robot_data, desired_position, desired_rotation, q_init, link)
        
#     def jacobian(self, q, link='ee'):
#         return jacobian(self.robot_model, self.robot_data, q, link)
    
#     def get_joint_pos(self, obs):
#         return obs[self.env_info['puck_pos_ids']]
    
#     def get_joint_pos(self, obs):
#         return obs[self.env_info['joint_pos_ids']]
    
#     def get_ee_pose(self, obs):
#         return self.forward_kinematics(self.get_joint_pos(obs))
    
#     def get_opponent_ee_pose(self, obs):
#         return obs[self.env_info['opponent_ee_ids']]




if __name__ == '__main__':
    
    env = AirHockeyChallengeWrapper(env="7dof-hit", interpolation_order=3)

    agent1 = PPOAgent(env.env_info, agent_id=1)
    obs = env.reset()
    agents = agent1

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
