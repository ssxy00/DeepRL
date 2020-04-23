# -*- coding: utf-8 -*-
# @Time        : 2020/4/19 17:52
# @Author      : ssxy00
# @File        : model.py
# @Description :

import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.nn.init import normal_


class FFN(nn.Module):
    """
    state -> pi(a|s), V(s)
    """
    def __init__(self, model_config):
        super(FFN, self).__init__()
        self.state_space_size = model_config.state_space_size
        self.action_space_size = model_config.action_space_size

        # initialize model
        self.linear_1 = nn.Linear(self.state_space_size, model_config.mid_dim)
        self.activation_1 = F.relu
        self.value_head = nn.Linear(model_config.mid_dim, 1)
        self.policy_head = nn.Linear(model_config.mid_dim, self.action_space_size)

        if model_config.init_weight:
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
        hidden_states = self.activation_1(self.linear_1(state))
        # compute value
        value = self.value_head(hidden_states)
        # compute policy distribution \pi(a|s)
        dist = Categorical(logits=self.policy_head(hidden_states))
        return value, dist
