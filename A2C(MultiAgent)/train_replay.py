# -*- coding: utf-8 -*-
# @Time        : 2020/5/12 21:56
# @Author      : ssxy00
# @File        : train_replay.py
# @Description :

import argparse
import torch
import random
import numpy as np
import gfootball.env as football_env

from model import Actor, Critic
from maa2c_replay import MAA2C



def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def main(args):
    print(args.early_stop)
    print(args.disable_actions)
    set_seed(args.seed)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # initialize environment
    n_players = 3
    env = football_env.create_environment(env_name="academy_3_vs_1_with_keeper",
                                          representation="simple115",
                                          number_of_left_players_agent_controls=n_players,
                                          stacked=False,
                                          logdir="/tmp/football",
                                          write_goal_dumps=False,
                                          write_full_episode_dumps=False,
                                          render=False)

    # state and action space
    state_space_size = env.observation_space.shape[1]  # we are using simple115 representation
    action_space_size = env.action_space.nvec.tolist()[0]  # 三个 players 动作空间相同
    action_list = list(range(action_space_size))
    # state[98:100] 表示控制的三个球员
    if args.disable_actions:
        action_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        action_space_size = len(action_list)

    # model
    print("loading models")
    actors = [Actor(state_space_size=state_space_size, action_space_size=action_space_size) for _ in range(n_players)]
    critic = Critic(state_space_size=state_space_size)

    # maa2c
    maa2c = MAA2C(args=args, env=env, actors=actors, critic=critic, action_list=action_list, device=device)
    print("learn")
    maa2c.learn()


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str,
                        help="path to save model")
    parser.add_argument("--log_dir", type=str,
                        help="path to save model")
    parser.add_argument("--seed", default=1, type=int, help="Random seed")
    parser.add_argument("--max_memory_size", default=20000, type=int)
    parser.add_argument("--episodes_before_training", default=100, type=int)
    parser.add_argument("--n_episodes", default=25000, type=int)
    parser.add_argument("--episode_max_length", default=300, type=int)
    parser.add_argument("--batch_size", default=256, type=int, help="Batch size")
    parser.add_argument("--save_interval", default=1000, type=int, help="save checkpoints interval")
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--gamma", default=0.99)
    parser.add_argument("--epsilon", default=0.1)
    parser.add_argument("--tau", default=0.01, type=float)
    parser.add_argument("--early_stop", action="store_true")
    parser.add_argument("--disable_actions", action="store_true")
    parser.add_argument("--entropy", action="store_true")
    parser.add_argument("--entropy_coef", default=0., type=float)

    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()
