# -*- coding: utf-8 -*-
# @Time        : 2020/5/12 22:05
# @Author      : ssxy00
# @File        : utils.py
# @Description :

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.autograd import Variable


class ConditionReplayMemory:
    def __init__(self, max_memory_size, device):
        self.succeed_memory = ReplayMemory(10000, device=device)
        self.other_memory = ReplayMemory(10000, device=device)

    def add_experience(self, states, actions, returns, reward_of_episode):
        if reward_of_episode == 1:
            self.succeed_memory.add_experience(states=states, actions=actions, returns=returns)
        else:
            self.other_memory.add_experience(states=states, actions=actions, returns=returns)

    def sample_minibatch(self, batch_size=16):
        success_batch_size = batch_size // 2
        if len(self.succeed_memory) < success_batch_size:
            success_batch_states, success_batch_actions, success_batch_returns = self.succeed_memory.return_all()
        else:
            success_batch_states, success_batch_actions, success_batch_returns = self.succeed_memory.sample_mini_batch(
                success_batch_size)

        other_batch_states, other_batch_actions, other_batch_returns = self.other_memory.sample_mini_batch(
            batch_size - success_batch_states.shape[0])

        return torch.cat((success_batch_states, other_batch_states), dim=0), \
               torch.cat((success_batch_actions, other_batch_actions), dim=0), \
               torch.cat((success_batch_returns, other_batch_returns), dim=0)


class ReplayMemory:
    def __init__(self, max_memory_size, device):
        """experience pool"""
        self.max_memory_size = max_memory_size
        self.device = device
        self.states_memory = torch.tensor([], device=device)  # n_experiences, state_space_size
        self.actions_memory = torch.tensor([], dtype=torch.long, device=device)  # n_experiences, n_players
        self.returns_memory = torch.tensor([], dtype=torch.float, device=device)  # n_experiences

    def __len__(self):
        return self.states_memory.shape[0]

    def add_experience(self, states, actions, returns):
        """
        :param states: episode_len, state_space_size
        :param actions: episode_len, n_players
        :param returns: episode_len
        :return:
        """

        self.states_memory = torch.cat([self.states_memory, states], dim=0)
        self.actions_memory = torch.cat([self.actions_memory, actions], dim=0)
        self.returns_memory = torch.cat([self.returns_memory, returns], dim=0)
        if self.states_memory.shape[0] > self.max_memory_size:
            self.states_memory = self.states_memory[-self.max_memory_size:, :]
            self.actions_memory = self.actions_memory[-self.max_memory_size:, :]
            self.returns_memory = self.returns_memory[-self.max_memory_size:]

    def sample_mini_batch(self, batch_size=16):
        indices = np.random.choice(list(range(self.states_memory.shape[0])), size=batch_size, replace=False)
        indices = torch.tensor(indices, device=self.device)
        # indices = torch.tensor(np.random.randint(0, self.states_memory.shape[0], size=batch_size))
        batch_states = self.states_memory.index_select(0, indices)
        batch_actions = self.actions_memory.index_select(0, indices)
        batch_returns = self.returns_memory.index_select(0, indices)

        return batch_states, batch_actions, batch_returns

    def return_all(self):
        return self.states_memory, self.actions_memory, self.returns_memory



def process_state(state):
    # 将 env 返回的每个 agent 的 state 整合成一个 state
    processed_state = state[0]
    processed_state[98:101] = 1
    return processed_state
