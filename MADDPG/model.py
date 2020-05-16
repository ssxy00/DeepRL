# -*- coding: utf-8 -*-
# @Time        : 2020/5/12 21:54
# @Author      : ssxy00
# @File        : model.py
# @Description :

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_


class Actor(nn.Module):

    def __init__(self, state_space_size, action_space_size, mid_dim=60, init_weight=True):
        super(Actor, self).__init__()
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.mid_dim = mid_dim

        self.linear_1 = nn.Linear(state_space_size, mid_dim)
        self.activation_1 = F.relu
        self.linear_2 = nn.Linear(mid_dim, action_space_size)

        if init_weight:
            self._reset_parameters()

    def _reset_parameters(self):
        r"""Initiate parameters in the model."""
        for p in self.parameters():
            normal_(p, 0., 0.1)

    def forward(self, state):
        """
        :param state:
        :return:
        """
        logits = self.linear_1(state)
        logits = self.activation_1(logits)
        logits = self.linear_2(logits)
        return logits

class Critic(nn.Module):
    """
    state -> pi(a|s), V(s)
    """
    def __init__(self, state_space_size, action_space_size, n_players, state_mid_dim=60, action_mid_dim=2,
                 mid_dim_2=30, init_weight=True):
        super(Critic, self).__init__()
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.n_players = n_players

        self.state_linear_1 = nn.Linear(state_space_size, state_mid_dim)
        self.action_linear_1 = nn.Linear(action_space_size, action_mid_dim)
        self.activation_1 = F.relu
        self.linear_2 = nn.Linear(state_mid_dim + n_players * action_mid_dim, mid_dim_2)
        self.activation_2 = F.relu
        self.linear_3 = nn.Linear(mid_dim_2, 1)

        if init_weight:
            self._reset_parameters()

    def _reset_parameters(self):
        r"""Initiate parameters in the model."""
        for p in self.parameters():
            normal_(p, 0., 0.1)

    def forward(self, state, actions):
        """
        :param state: bsz, dim_state
        :param actions: bsz, n_players, dim_action
        :return:
        """
        state_hidden_states = self.state_linear_1(state)
        action_hidden_states = self.action_linear_1(actions)
        hidden_states = torch.cat([state_hidden_states,
                                   action_hidden_states.view(action_hidden_states.shape[0], -1)], dim=1)
        logits = self.linear_3(self.activation_2(self.linear_2(self.activation_1(hidden_states))))
        return logits


