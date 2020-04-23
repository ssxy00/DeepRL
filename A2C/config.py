# -*- coding: utf-8 -*-
# @Time        : 2020/4/19 17:46
# @Author      : ssxy00
# @File        : config.py
# @Description :

class FFNModelConfig:
    def __init__(self, state_space_size, action_space_size):
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.mid_dim = 60
        self.init_weight = True

class A2CConfig:
    def __init__(self):
        # train
        self.seed = 0
        self.lr = 5e-4
        self.max_episodes = 20000
        self.device = 'cpu'
        self.save_dir = "./checkpoints"  # 保存 checkpoints 的路径
        self.save_interval = 1000  # 每隔多少个 episodes 存储 checkpoint
        self.log_dir = "./logs"  # 输出 tensorboard log 的路径
        self.k = 500 # average mean window size
        # self.eval_interval = 100
        # a2c
        self.n_steps = 200 # 够不够？ # academy_empty_goal_close 32
        self.n_envs = 1 # TODO multiprocessing
        self.critic_coef = 0.5
        self.entropy_coef = 0.
        self.gamma = 0.99
        self.forbid_actions = False
        self.env_name = "academy_empty_goal"
