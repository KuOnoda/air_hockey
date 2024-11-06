from copy import deepcopy

from air_hockey_challenge.constraints import *
from air_hockey_challenge.environments import position_control_wrapper as position
from air_hockey_challenge.utils.kinematics import forward_kinematics, inverse_kinematics, jacobian
from air_hockey_challenge.utils import robot_to_world
from mushroom_rl.core import Environment

from gym import spaces


class AirHockeyChallengeWrapper(Environment):
    def __init__(self, env, custom_reward_function=None, interpolation_order=3, **kwargs):
        """
        Environment Constructor

        Args:
            env [string]:
                The string to specify the running environments. Available environments: [3dof-hit, 3dof-defend].
                [7dof-hit, 7dof-defend, 7dof-prepare, tournament] will be available once the corresponding stage starts.
            custom_reward_function [callable]:
                You can customize your reward function here.
            interpolation_order (int, 3): Type of interpolation used, has to correspond to action shape. Order 1-5 are
                    polynomial interpolation of the degree. Order -1 is linear interpolation of position and velocity.
                    Set Order to None in order to turn off interpolation. In this case the action has to be a trajectory
                    of position, velocity and acceleration of the shape (20, 3, n_joints)
        """

        env_dict = {
            "tournament": position.IiwaPositionTournament,

            "7dof-hit": position.IiwaPositionHit,
            "7dof-defend": position.IiwaPositionDefend,
            "7dof-prepare": position.IiwaPositionPrepare,

            "3dof-hit": position.PlanarPositionHit,
            "3dof-defend": position.PlanarPositionDefend
        }

        if env == "tournament" and type(interpolation_order) != tuple:
            interpolation_order = (interpolation_order, interpolation_order)

        self.base_env = env_dict[env](interpolation_order=interpolation_order, **kwargs)
        self.env_name = env
        self.env_info = self.base_env.env_info

        self.observation_space = spaces.Box(low=-1, high=1, shape=(23,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        if custom_reward_function:
            self.base_env.reward = lambda state, action, next_state, absorbing: custom_reward_function(self.base_env,
                                                                                                       state, action,
                                                                                                       next_state,
                                                                                                       absorbing)
        self.robot_model = copy.deepcopy(self.env_info['robot']['robot_model'])
        self.robot_data = copy.deepcopy(self.env_info['robot']['robot_data'])
        self.sim_dt = self.env_info['dt']

        # Get the initial observation and process it to set current_ee_pos and current_joint_pos
        initial_obs = self.base_env.reset()
        self.process_raw_obs(initial_obs)

        constraint_list = ConstraintList()
        constraint_list.add(JointPositionConstraint(self.env_info))
        constraint_list.add(JointVelocityConstraint(self.env_info))
        constraint_list.add(EndEffectorConstraint(self.env_info))
        if "7dof" in self.env_name or self.env_name == "tournament":
            constraint_list.add(LinkConstraint(self.env_info))

        self.env_info['constraints'] = constraint_list
        self.env_info['env_name'] = self.env_name

        self.robot_joint_pos_limit: np.ndarray = self.env_info["robot"][
            "joint_pos_limit"
        ]
        self.robot_joint_vel_limit: np.ndarray = self.env_info["robot"][
            "joint_vel_limit"
        ]

        super().__init__(self.base_env.info)

    def step(self, action):

        # actionを関節角と速度に変換
        action = self.process_raw_act(action)

        obs, reward, done, info = self.base_env.step(action)

        obs = self.process_raw_obs(obs)

        if "tournament" in self.env_name:
            info["constraints_value"] = list()
            info["jerk"] = list()
            for i in range(2):
                obs_agent = obs[i * int(len(obs) / 2): (i + 1) * int(len(obs) / 2)]
                info["constraints_value"].append(deepcopy(self.env_info['constraints'].fun(
                    obs_agent[self.env_info['joint_pos_ids']], obs_agent[self.env_info['joint_vel_ids']])))
                info["jerk"].append(
                    self.base_env.jerk[i * self.env_info['robot']['n_joints']:(i + 1) * self.env_info['robot'][
                        'n_joints']])

            info["score"] = self.base_env.score
            info["faults"] = self.base_env.faults

        else:
            info["constraints_value"] = deepcopy(self.env_info['constraints'].fun(obs[self.env_info['joint_pos_ids']],
                                                                                  obs[self.env_info['joint_vel_ids']]))
            info["jerk"] = self.base_env.jerk
            info["success"] = self.check_success(obs)

        return obs, reward, done, info

    def render(self, record=False):
        return self.base_env.render(record=record)

    def reset(self, state=None):
        return self.base_env.reset(state)

    def check_success(self, obs):
        puck_pos, puck_vel = self.base_env.get_puck(obs)

        puck_pos, _ = robot_to_world(self.base_env.env_info["robot"]["base_frame"][0], translation=puck_pos)
        success = 0

        if "hit" in self.env_name:
            if puck_pos[0] - self.base_env.env_info['table']['length'] / 2 > 0 and \
                    np.abs(puck_pos[1]) - self.base_env.env_info['table']['goal_width'] / 2 < 0:
                success = 1

        elif "defend" in self.env_name:
            if -0.8 < puck_pos[0] <= -0.2 and puck_vel[0] < 0.1:
                success = 1

        elif "prepare" in self.env_name:
            if -0.8 < puck_pos[0] <= -0.2 and np.abs(puck_pos[1]) < 0.39105 and puck_vel[0] < 0.1:
                success = 1
        return success
    
    def process_raw_obs(self, obs: np.ndarray) -> np.ndarray:
        """obsの前処理（正規化）"""
        #現在の手先位置
        self.current_ee_pos = self.get_ee_pose(obs)[0]
        self.current_joint_pos = self.get_joint_pos(obs)

        # current_joint_pos_normalized = np.clip(
        #     self._normalize_value(
        #         self.current_joint_pos,
        #         low_in=self.robot_joint_pos_limit[0, :],
        #         high_in=self.robot_joint_pos_limit[1, :],
        #     ),
        #     -1.0,
        #     1.0,
        # )

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
        # # Limit the joint velocities to the maximum allowed
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



if __name__ == "__main__":
    env = AirHockeyChallengeWrapper(env="7dof-hit")
    env.reset()

    R = 0.
    J = 0.
    gamma = 1.
    steps = 0
    while True:
        action = np.random.uniform(-1, 1, (2, env.env_info['robot']['n_joints'])) * 3
        observation, reward, done, info = env.step(action)
        print(env.env_info['joint_pos_ids'])

        print(action.shape) # (2,7)
        print(observation.shape) # (23,)
        
        env.render()
        gamma *= env.info.gamma
        J += gamma * reward
        R += reward
        steps += 1
        if done or steps > env.info.horizon:
            print("J: ", J, " R: ", R)
            R = 0.
            J = 0.
            gamma = 1.
            steps = 0
            env.reset()
