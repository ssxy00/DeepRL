# -*- coding: utf-8 -*-
# @Time        : 2020/4/19 17:50
# @Author      : ssxy00
# @File        : train.py
# @Description :

from tqdm import tqdm
import torch
import random
import numpy as np
import gfootball.env as football_env

from model import FFN
from a2c import A2C
from config import FFNModelConfig, A2CConfig

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def main():
    a2c_config = A2CConfig()
    set_seed(a2c_config.seed)

    # initialize environment
    env = football_env.create_environment(env_name=a2c_config.env_name,
                                          representation="simple115",
                                          number_of_left_players_agent_controls=1,
                                          stacked=False,
                                          logdir="/tmp/football",
                                          write_goal_dumps=False,
                                          write_full_episode_dumps=False,
                                          render=False)



    # state and action space
    state_space_size = env.observation_space.shape[0]  # we are using simple115 representation
    if a2c_config.forbid_actions:
        action_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 15]  # forbid some actions
        action_space_size = len(action_list)
    else:
        action_list = list(range(env.action_space.n))  # default action space
        action_space_size = len(action_list)

    # initialize model
    model_config = FFNModelConfig(state_space_size=state_space_size, action_space_size=action_space_size)
    model = FFN(model_config)

    # TODO multiprocessing env
    a2c = A2C(env=env, model=model, a2c_config=a2c_config, action_list=action_list)
    a2c.learn()

if __name__ == "__main__":
    main()