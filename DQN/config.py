# -*- coding: utf-8 -*-
# @Time        : 2020/4/3 17:28
# @Author      : ssxy00
# @File        : config.py
# @Description :


class FFNModelConfig:
    def __init__(self, state_space_size, action_space_size):
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.mid_dim = 60


class DQNConfig:
    def __init__(self):
        # train
        self.seed = 0
        self.batch_size = 32
        self.lr = 1e-3
        self.max_episodes = 10000
        self.device = 'cpu'
        self.save_dir = "./checkpoints" # 保存 checkpoints 的路径
        self.save_interval = 1000 # 每隔多少个 episodes 存储 checkpoint
        self.log_dir = "./logs" # 输出 tensorboard log 的路径
        self.eval_interval = 100
        # dqn
        self.gamma = 0.99
        self.max_epsilon = 1
        self.min_epsilon = 0.1
        self.decay_episodes = 1000
        self.update_nums = 100  # 每次 update 的次数
        self.target_update_episodes = 100
        self.max_memory_size = 20000
        self.forbid_actions = False
        self.replay_memory_warmup = 10
