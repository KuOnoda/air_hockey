import ray
from ray import tune
from ray.rllib.algorithm.ppo import PPOConfig
from ray.rllib.env import ExternalEnv
from air_hockey_challenge.framework.air_hockey_challenge_wrapper import AirHockeyChallengeWrapper

# Rayの初期化
ray.init()

# カスタム環境の設定（Gym形式にする）
class AirHockeyEnv(ExternalEnv):
    def __init__(self, env_config):
        super().__init__()
        self.env = AirHockeyChallengeWrapper(env="7dof-hit", interpolation_order=3, debug=True)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

    def reset(self):
        return self.env.reset()

# 環境の登録
tune.register_env("AirHockeyEnv", lambda config: AirHockeyEnv(config))

# PPOエージェントの設定
config = {
    "env": "AirHockeyEnv",
    "num_workers": 1,  # 並列ワーカー数
    "framework": "torch",  # PyTorchで実装
    "train_batch_size": 4000,
    "sgd_minibatch_size": 128,
    "num_sgd_iter": 10,
    "lr": 1e-4,
    "model": {
        "fcnet_hiddens": [256, 256],
        "fcnet_activation": "relu",
    },
}

# PPOエージェントの訓練
trainer = PPOTrainer(config=config)

# 訓練ループ
for i in range(100):
    result = trainer.train()
    print(f"Iteration {i}: episode_reward_mean = {result['episode_reward_mean']}")

    if i % 10 == 0:
        checkpoint = trainer.save()
        print(f"Checkpoint saved at {checkpoint}")

# 訓練済みエージェントの評価
env = AirHockeyChallengeWrapper(env="7dof-hit", interpolation_order=3, debug=True)
obs = env.reset()

while True:
    action = trainer.compute_action(obs)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()

# Rayの終了
ray.shutdown()
