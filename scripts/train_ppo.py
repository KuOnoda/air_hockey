# import sys
# print("Python executable path:", sys.executable)
# print("Python version:", sys.version)




import gym

import numpy as np

import copy

from air_hockey_challenge.framework import AirHockeyChallengeWrapper
from air_hockey_challenge.utils.kinematics import forward_kinematics, inverse_kinematics, jacobian

import torch
import torch.nn as nn

from mushroom_rl.algorithms.actor_critic import PPO
from mushroom_rl.core import Core
from mushroom_rl.core.environment import MDPInfo
from mushroom_rl.core.environment import Environment
from mushroom_rl.policy import GaussianPolicy
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.parameters import Parameter
from mushroom_rl.utils.dataset import compute_J
from mushroom_rl.algorithms.actor_critic import PPO
from mushroom_rl.core import Agent
from mushroom_rl.environments import Gym
from mushroom_rl.policy import GaussianPolicy
from mushroom_rl.policy import DeterministicPolicy
from mushroom_rl.utils.parameters import Parameter
from mushroom_rl.utils.optimizers import AdaptiveOptimizer
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import LinearApproximator
from mushroom_rl.policy import OrnsteinUhlenbeckPolicy
from mushroom_rl.policy import TorchPolicy
from mushroom_rl.policy import GaussianTorchPolicy


import torch
import torch.nn as nn
import torch.optim as optim

# 環境の初期化
env = AirHockeyChallengeWrapper(env="7dof-hit", interpolation_order=3)

input_shape = env.observation_space.shape
output_shape = env.action_space.shape

print(input_shape, output_shape)

class PPOPolicyNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, use_cuda=False, dropout=0.0):
        super(PPOPolicyNetwork, self).__init__()
        n_input = input_shape[-1]
        n_output = output_shape[-1]
        print(output_shape)
        self.fc1 = nn.Linear(n_input, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, n_output)

        self.use_cuda = use_cuda
        self.dropout = dropout

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

policy_network = PPOPolicyNetwork
policy_optimizer = {'class': optim.Adam, 'params': {'lr': 3e-4}}
policy = GaussianTorchPolicy(policy_network, input_shape, output_shape)\

critic_params = dict(network=PPOPolicyNetwork,
                    optimizer={'class': optim.Adam,
                                'params': {'lr': 3e-4}},
                    loss=nn.MSELoss(),
                    input_shape=input_shape,
                    output_shape=(1,))


class PPO_Agent(PPO):
    """mushroom-rlのPPOを継承してエージェントに対する関数を実装"""

    def __init__(self, mdp_info, env_info, policy, actor_optimizer, critic_params, n_epochs_policy, batch_size, eps_ppo,  lam, ent_coeff=0.0):
        super().__init__(mdp_info, policy, actor_optimizer, critic_params, n_epochs_policy, batch_size, eps_ppo, lam, ent_coeff=0.0)
        self.env_info = env_info
        self.robot_model = copy.deepcopy(env_info['robot']['robot_model'])
        self.robot_data = copy.deepcopy(env_info['robot']['robot_data'])

        self.sim_dt = self.env_info['dt']

    def draw_action(self, obs: np.ndarray) -> np.ndarray:
        """PPOの継承元のAgentクラスのdraw_actionをオーバーライドして、process_raw_actをかます"""
        obs = self.process_raw_obs(obs)

        if self.phi is not None:
            obs = self.phi(obs)

        if self.next_action is None:
            action = self.policy.draw_action(obs)
            return action
            return self.process_raw_act(action)
        else:
            action = self.next_action
            self.next_action = None   
            return action 
            return self.process_raw_act(action)

    def process_raw_obs(self, obs: np.ndarray) -> np.ndarray:
        """obsの前処理（正規化）"""
        #現在の手先位置
        self.current_ee_pos = self.get_ee_pose(obs)[0]
        self.current_joint_pos = self.get_joint_pos(obs)

        return obs

    def process_raw_act(self, action: np.ndarray) -> np.ndarray:
        """actの前処理（ターゲット位置(2,)をアームの関節角と速度に変換（2,7））"""
        #目標の手先位置
        target_ee_pos_xy = action
        target_ee_pos = np.array([target_ee_pos_xy[0], target_ee_pos_xy[1], 0.5], dtype=target_ee_pos_xy.dtype)

        #print(target_ee_pos, self.current_ee_pos)

        target_ee_disp = target_ee_pos - self.current_ee_pos

        jac = self.jacobian(self.current_joint_pos)[:3]
        jac_pinv = np.linalg.pinv(jac)
        s = np.linalg.svd(jac, compute_uv=False)
        s[:2] = np.mean(s[:2])
        # s[2] *= self.z_position_control_tolerance
        s[2] *= 0.5
        s = 1 / s
        s = s / np.sum(s)
        joint_disp = jac_pinv * target_ee_disp
        joint_disp = np.average(joint_disp, axis=1, weights=s)

        # Convert to joint velocities based on joint displacements
        joint_vel = joint_disp / self.sim_dt

        ## 安全のために制約を課す部分
        # Limit the joint velocities to the maximum allowed
        # joints_below_vel_limit = joint_vel < self.robot_joint_vel_limit_scaled[0, :]
        # joints_above_vel_limit = joint_vel > self.robot_joint_vel_limit_scaled[1, :]
        # joints_outside_vel_limit = np.logical_or(
        #     joints_below_vel_limit, joints_above_vel_limit
        # )
        # if np.any(joints_outside_vel_limit):
        #     downscaling_factor = 1.0
        #     for joint_i in np.where(joints_outside_vel_limit)[0]:
        #         downscaling_factor = min(
        #             downscaling_factor,
        #             1
        #             - (
        #                 (
        #                     joint_vel[joint_i]
        #                     - self.robot_joint_vel_limit_scaled[
        #                         int(joints_above_vel_limit[joint_i]), joint_i
        #                     ]
        #                 )
        #                 / joint_vel[joint_i]
        #             ),
        #         )
        #     # Scale down the joint velocities to the maximum allowed limits
        #     joint_vel *= downscaling_factor

        # Update the target joint positions based on joint velocities
        joint_pos = self.current_joint_pos + (self.sim_dt * joint_vel)

        # Assign the action
        return np.array([joint_pos, joint_vel])

    def get_joint_pos(self, obs):
        """obsから関節位置を取得"""
        return obs[self.env_info['joint_pos_ids']]

    def get_ee_pose(self, obs):
        """順運動学で手の位置を取得"""
        return forward_kinematics(mj_model=self.robot_model, mj_data=self.robot_data, q=self.get_joint_pos(obs),link='ee')

    def jacobian(self, q, link='ee'):
        """ヤコビアンを計算"""
        return jacobian(self.robot_model, self.robot_data, q, link)

# PPO エージェントの設定
agent = PPO_Agent(
    mdp_info=env.info,
    env_info=env.env_info,
    policy=policy,
    actor_optimizer=policy_optimizer,
    critic_params = critic_params,
    n_epochs_policy=4,
    batch_size=64,
    eps_ppo=0.2,
    lam=0.95,
    ent_coeff=0.0,
)

core = Core(agent, env)
n_episodes = 5000

for episode in range(n_episodes):
    core.learn(n_episodes=100, n_steps_per_fit=100, render=False)
    dataset = core.evaluate(n_episodes=100,)
    J = compute_J(dataset, gamma=0.9)
    print(f'Episode {episode + 1}: Return = {np.mean(J)}')


# 可視化
obs = env.reset()
done = False

while not done:
        action = agent.draw_action(obs)
        next_obs, reward, done, _ = env.step(action)
        obs = next_obs
        env.render()  # 環境をレンダリング

