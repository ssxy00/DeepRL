# -*- coding: utf-8 -*-
# @Time        : 2020/4/3 00:21
# @Author      : ssxy00
# @File        : dqn.py
# @Description : DQN algorithm with experience replay and fixed Q-target

import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from model import FFN
from gfootball.env import football_action_set
from tensorboardX import SummaryWriter


class DecayEpsilon:
    def __init__(self, max_epsilon, min_epsilon, decay_episodes):
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.decay_episodes = decay_episodes

    def get_epsilon(self, episode):
        if episode < self.decay_episodes:
            return self.max_epsilon - (self.max_epsilon - self.min_epsilon) * episode / self.decay_episodes
        else:
            return self.min_epsilon


class ReplayMemory:
    """experience pool"""
    def __init__(self, max_memory_size):
        self.max_memory_size = max_memory_size
        self.states_memory = torch.tensor([])
        self.actions_memory = torch.tensor([], dtype=torch.long)
        self.rewards_memory = torch.tensor([])
        self.new_states_memory = torch.tensor([])

    def __len__(self):
        return self.states_memory.shape[0]

    def add_experience(self, state, action, reward, new_state):
        self.states_memory = torch.cat([self.states_memory, torch.tensor([state])], dim=0)
        self.actions_memory = torch.cat([self.actions_memory, torch.tensor([action])], dim=0)
        self.rewards_memory = torch.cat([self.rewards_memory, torch.tensor([reward], dtype=torch.float)], dim=0)
        self.new_states_memory = torch.cat([self.new_states_memory, torch.tensor([new_state])], dim=0)
        if self.states_memory.shape[0] > self.max_memory_size:
            self.states_memory = self.states_memory[1:, :]
            self.actions_memory = self.actions_memory[1:]
            self.rewards_memory = self.rewards_memory[1:]
            self.new_states_memory = self.new_states_memory[1:, :]

    def sample_mini_batch(self, batch_size=16):
        # indices = np.random.choice(list(range(self.states_memory.shape[0])), size=batch_size, replace=False, p=((self.rewards_memory * 10).abs().softmax(-1).numpy()))
        # indices = torch.tensor(indices)
        indices = torch.tensor(np.random.randint(0, self.states_memory.shape[0], size=batch_size))
        batch_states = self.states_memory.index_select(0, indices)
        batch_actions = self.actions_memory.index_select(0, indices)
        batch_rewards = self.rewards_memory.index_select(0, indices)
        batch_new_states = self.new_states_memory.index_select(0, indices)
        return batch_states, batch_actions, batch_rewards, batch_new_states


class DQN:
    def __init__(self, env, model, dqn_config, action_list):
        self.env = env
        self.action_list = action_list
        self.action_space_size = len(action_list)

        self.model = model
        self.old_model = FFN(self.model.model_config)
        self.old_model.load_state_dict(self.model.state_dict())

        self.gamma = dqn_config.gamma
        self.epsilon_func = DecayEpsilon(max_epsilon=dqn_config.max_epsilon, min_epsilon=dqn_config.min_epsilon,
                                         decay_episodes=dqn_config.decay_episodes)
        self.target_update_episodes = dqn_config.target_update_episodes

        self.replay_memory = ReplayMemory(dqn_config.max_memory_size)
        self.replay_memory_warmup = dqn_config.replay_memory_warmup

        # training config

        self.device = torch.device(dqn_config.device) if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)
        self.old_model.to(self.device)
        self.max_episodes = dqn_config.max_episodes
        self.batch_size = dqn_config.batch_size
        self.optimizer = Adam(self.model.parameters(), lr=dqn_config.lr, weight_decay=0.01)
        self.criterion = nn.MSELoss()
        self.save_dir = dqn_config.save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.save_interval = dqn_config.save_interval
        self.update_nums = dqn_config.update_nums
        self.eval_interval = dqn_config.eval_interval

        # log
        self.writer = SummaryWriter(dqn_config.log_dir)
        if not os.path.exists(dqn_config.log_dir):
            os.makedirs(dqn_config.log_dir)

    def select_action(self, state, epsilon=0):
        """
        with prob \epsilon select a random action, otherwise select action with max Q-value
        """
        if np.random.random_sample() < epsilon:
            action = np.random.randint(0, self.action_space_size)
        else:
            state_tensor = torch.tensor([state], device=self.device)
            logits = self.model(state_tensor)
            action = logits.argmax(-1).item()
        return action

    def update(self):
        batch_states, batch_actions, batch_rewards, batch_new_states = self.replay_memory.sample_mini_batch(
            self.batch_size)
        predict_logits = self.model(batch_states.to(self.device))  # bsz, action_space_size
        predict_values = predict_logits.gather(1, batch_actions.to(self.device).unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            target_logits = self.old_model(batch_new_states.to(self.device))
            target_values = target_logits.max(1)[0] * self.gamma
            target_values += batch_rewards.to(self.device)

        loss = self.criterion(target_values, predict_values)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()

    def train(self):
        self.model.train()
        self.old_model.eval()
        # 所有 episodes 的平均
        mean_reward = 0
        mean_loss = 0
        mean_steps = 0  # 当前所有 episodes 平均 step 数

        print("collecting experience")
        state = self.env.reset()
        count = 1
        while len(self.replay_memory) < self.replay_memory_warmup:
            # 选择容易完成进球的动作来快速收集非 0 reward
            action = np.random.choice([4, 5, 6, 12])
            obs, rew, done, info = self.env.step(action)
            if rew == 1 or rew == -1:
                count += 1
                self.replay_memory.add_experience(state=state, action=action, reward=rew, new_state=obs)
            state = obs
            if done:
                state = self.env.reset()

        print("training")

        tqdm_episodes = tqdm(range(1, self.max_episodes + 1))  # episode 从 1 开始计数
        for episode in tqdm_episodes:
            state = self.env.reset()
            done = False
            steps = 0  # 当前 episode 的 step 数

            while not done:
                epsilon = self.epsilon_func.get_epsilon(episode)
                action = self.select_action(state, epsilon)
                obs, rew, done, info = self.env.step(self.action_list[action])
                steps += 1
                self.replay_memory.add_experience(state=state, action=action, reward=rew, new_state=obs)
                state = obs

            mean_reward = (mean_reward * (episode - 1) + rew) / episode
            mean_steps = (mean_steps * (episode - 1) + steps) / episode

            # update model parameters with adam
            loss = 0
            self.model.train()
            # 在一个 episode 结束后做 update_nums 次更新
            for _ in range(self.update_nums):
                loss += self.update()
            loss /= self.update_nums
            mean_loss = (mean_loss * (episode - 1) + loss) / episode
            tqdm_episodes.set_postfix({'episode': episode,
                                       'epsilon': epsilon,
                                       'mean_steps': mean_steps,
                                       'mean_reward': mean_reward,
                                       'mean_loss': mean_loss})
            self.writer.add_scalar('Train/epsilon', epsilon, episode)
            self.writer.add_scalar('Train/steps', steps, episode)
            self.writer.add_scalar('Train/mean_steps', mean_steps, episode)
            self.writer.add_scalar('Train/reward', rew, episode)
            self.writer.add_scalar('Train/mean_reward', mean_reward, episode)
            self.writer.add_scalar('Train/loss', loss, episode)
            self.writer.add_scalar('Train/mean_loss', mean_loss, episode)

            # reset old model
            if episode % self.target_update_episodes == 0:
                self.old_model.load_state_dict(self.model.state_dict())

            if episode % self.eval_interval == 0:
                self.eval(episode)

            # save checkpoint
            if episode % self.save_interval == 0:
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, f"checkpoint{episode}.pt"))



    def eval(self, episode):
        """以当前模型的参数进行 100 个 episodes 的游戏，以 e-greedy 选择动作，计算 100 个 episodes 的平均
         reward 和 steps 作为当前训练 episode 的评估参考"""
        self.model.eval()
        mean_steps = 0
        mean_reward = 0
        epsilon = 0.05
        n_episodes = 100

        for idx in range(n_episodes):
            steps = 0
            state = self.env.reset()
            # 记录最后一个 episode 的动作序列
            if idx == n_episodes - 1:
                actions = []
            done = False
            while not done:
                action = self.select_action(state, epsilon=epsilon)
                if idx == n_episodes - 1:
                    actions.append(football_action_set.named_action_from_action_set(self.env.unwrapped._env._action_set,
                                                                       self.action_list[action]))
                steps += 1
                obs, rew, done, info = self.env.step(self.action_list[action])
                # print(obs[94:97])
                state = obs
            # print(rew)
            mean_steps = (mean_steps * idx + steps) / (idx + 1)
            mean_reward = (mean_reward * idx + rew) / (idx + 1)

        print(f"\nepisode {episode}, mean steps {mean_steps}, mean reward {mean_reward}")
        # 打印最后一个 episode 的状态序列
        print(f"last episode action sequence:")
        print(' '.join([f"{action_i}" for action_i in actions]))
        self.writer.add_scalar('Eval/mean_steps', mean_steps, episode)
        self.writer.add_scalar('Eval/mean_reward', mean_reward, episode)

