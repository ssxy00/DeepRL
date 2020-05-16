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

class ReplayMemory:
    def __init__(self, max_memory_size):
        """experience pool"""
        self.max_memory_size = max_memory_size
        self.states_memory = torch.tensor([]) # n_experiences, state_space_size
        self.actions_memory = torch.tensor([], dtype=torch.float) # n_experiences, n_players, state_space_size
        self.rewards_memory = torch.tensor([]) # n_experiences
        self.new_states_memory = torch.tensor([]) # n_experiences, state_space_size
        self.episode_dones_memory = torch.tensor([], dtype=torch.float)
        self.next_idx = 0

    def __len__(self):
        return self.states_memory.shape[0]

    def add_experience(self, state, actions, reward, new_state, done):
        if self.states_memory.shape[0] < self.max_memory_size:
            self.states_memory = torch.cat([self.states_memory, torch.tensor([state])], dim=0)
            self.actions_memory = torch.cat([self.actions_memory, torch.tensor([actions], dtype=torch.float)], dim=0)
            self.rewards_memory = torch.cat([self.rewards_memory, torch.tensor([reward], dtype=torch.float)], dim=0)
            self.new_states_memory = torch.cat([self.new_states_memory, torch.tensor([new_state])], dim=0)
            self.episode_dones_memory = torch.cat([self.episode_dones_memory, torch.tensor([done], dtype=torch.float)], dim=0)
        else:
            self.states_memory[self.next_idx, :] = torch.tensor(state)
            self.actions_memory[self.next_idx, :, :] = torch.tensor(actions)
            self.rewards_memory[self.next_idx] = torch.tensor(reward)
            self.new_states_memory[self.next_idx, :] = torch.tensor(new_state)
            self.episode_dones_memory[self.next_idx] = torch.tensor(done)

    def sample_mini_batch(self, batch_size=16):
        # indices = np.random.choice(list(range(self.states_memory.shape[0])), size=batch_size, replace=False, p=((self.rewards_memory * 10).abs().softmax(-1).numpy()))
        # indices = torch.tensor(indices)
        indices = torch.tensor(np.random.randint(0, self.states_memory.shape[0], size=batch_size))
        batch_states = self.states_memory.index_select(0, indices)
        batch_actions = self.actions_memory.index_select(0, indices)
        batch_rewards = self.rewards_memory.index_select(0, indices)
        batch_new_states = self.new_states_memory.index_select(0, indices)
        batch_dones = self.episode_dones_memory.index_select(0, indices)
        return batch_states, batch_actions, batch_rewards, batch_new_states, batch_dones

# modified from https://github.com/shariqiqbal2810/maddpg-pytorch/blob/40388d7c18e4662cf23c826d97e209df9003d86c/utils/misc.py#L77
def soft_update(old_model, current_model, tau):
    for old_param, current_param in zip(old_model.parameters(), current_model.parameters()):
        old_param.data.copy_(old_param.data * (1.0 - tau) + current_param.data * tau)

# modified from https://github.com/shariqiqbal2810/maddpg-pytorch/blob/40388d7c18e4662cf23c826d97e209df9003d86c/utils/misc.py#L77
def convert_to_onehot(logits, epsilon=0.0):
    # TODO explore
    device = logits.device
    logits_shape = logits.shape
    logits = logits.view(-1, logits_shape[-1])
    one_hot = (logits == logits.max(1, keepdim=True)[0]).float()
    if epsilon == 0.:
        return one_hot.view(logits_shape)
    random_indices = np.random.choice(logits.shape[1], size=logits.shape[0])
    random_one_hot = torch.eye(logits.shape[1], device=device)[random_indices]
    explore_dist = Categorical(torch.tensor([1 - epsilon, epsilon], device=device))
    explore = explore_dist.sample([logits.shape[0], 1]).float()
    one_hot = one_hot * (1 - explore) + random_one_hot * explore

    return one_hot.view(logits_shape)

# copy from https://github.com/shariqiqbal2810/maddpg-pytorch/blob/40388d7c18e4662cf23c826d97e209df9003d86c/utils/misc.py#L77
def sample_gumbel(shape, device, eps=1e-20, tens_type=torch.FloatTensor):
    """Sample from Gumbel(0, 1)"""
    U = Variable(tens_type(*shape).uniform_(), requires_grad=False)
    U = U.to(device)
    return -torch.log(-torch.log(U + eps) + eps)

# copy from https://github.com/shariqiqbal2810/maddpg-pytorch/blob/40388d7c18e4662cf23c826d97e209df9003d86c/utils/misc.py#L77
def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(logits.shape, device=logits.device, tens_type=type(logits.data))
    return F.softmax(y / temperature, dim=1)

# copy from https://github.com/shariqiqbal2810/maddpg-pytorch/blob/40388d7c18e4662cf23c826d97e209df9003d86c/utils/misc.py#L77
def gumbel_softmax(logits, temperature=1.0, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        y_hard = convert_to_onehot(y)
        y = (y_hard - y).detach() + y
    return y


