# -*- coding: utf-8 -*-
# @Time        : 2020/5/12 21:56
# @Author      : ssxy00
# @File        : train.py
# @Description :

import argparse
import torch
import random
import numpy as np
import gfootball.env as football_env

from model import Actor, Critic
from maddpg import MADDPG


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def main(args):
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
    action_space_size = env.action_space.nvec.tolist()[0] # 三个 players 动作空间相同
    # state[98:100] 表示控制的三个球员

    # model
    print("loading models")
    actors = [Actor(state_space_size=state_space_size, action_space_size=action_space_size) for _ in range(n_players)]
    critics = [Critic(state_space_size=state_space_size, action_space_size=action_space_size, n_players=n_players) for _ in range(n_players)]
    old_actors = [Actor(state_space_size=state_space_size, action_space_size=action_space_size) for _ in range(n_players)]
    old_critics = [Critic(state_space_size=state_space_size, action_space_size=action_space_size, n_players=n_players) for _ in range(n_players)]
    for old_actor, actor in zip(old_actors, actors):
        old_actor.load_state_dict(actor.state_dict())
    for old_critic, critic in zip(old_critics, critics):
        old_critic.load_state_dict(critic.state_dict())

    # maddpg
    maddpg = MADDPG(env=env, action_list=list(range(action_space_size)), actors=actors, critics=critics,
                    old_actors=old_actors, old_critics=old_critics, args=args, device=device)
    print("learn")
    maddpg.learn()

def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="", type=str,
                        help="path to save model")
    parser.add_argument("--log_dir", default="", type=str,
                        help="path to save model")
    parser.add_argument("--seed", default=1, type=int, help="Random seed")
    parser.add_argument("--max_memory_size", default=20000, type=int)
    parser.add_argument("--episodes_before_training", default=100, type=int)
    parser.add_argument("--n_episodes", default=10000, type=int)
    parser.add_argument("--episode_max_length", default=300, type=int)
    parser.add_argument("--batch_size", default=256, type=int, help="Batch size")
    parser.add_argument("--save_interval", default=1000, type=int, help="save checkpoints interval")
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--gamma", default=0.99)
    parser.add_argument("--epsilon", default=0.1)
    parser.add_argument("--tau", default=0.01, type=float)


    args = parser.parse_args()
    main(args)

if __name__ == "__main__":
    cli_main()