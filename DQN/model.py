# -*- coding: utf-8 -*-
# @Time        : 2020/4/3 00:21
# @Author      : ssxy00
# @File        : model.py
# @Description :

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_

class FFN(nn.Module):
    """
    state -> action
    """
    def __init__(self, model_config):
        super(FFN, self).__init__()
        self.model_config = model_config
        self.state_space_size = model_config.state_space_size
        self.action_space_size = model_config.action_space_size

        # initialize model
        self.linear_1 = nn.Linear(self.state_space_size, self.model_config.mid_dim)
        self.activation_1 = F.relu
        self.linear_2 = nn.Linear(self.model_config.mid_dim, self.action_space_size)


    def forward(self, state):
        """
        :param state:
        :return:
        """
        x = self.activation_1(self.linear_1(state))
        x = self.linear_2(x)
        return x


