# -*- coding: utf-8 -*-
# @Time        : 2020/5/15 16:39
# @Author      : ssxy00
# @File        : maa2c_replay.py
# @Description :

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from gfootball.env import football_action_set
from utils import process_state, ConditionReplayMemory


class MAA2C:
    def __init__(self, args, env, actors, critic, action_list, device):
        self.device = device
        self.args = args
        self.env = env
        self.action_list = action_list
        self.action_space_size = len(action_list)

        self.actors = [actor.to(device) for actor in actors]
        self.critic = critic.to(device)
        self.n_players = len(self.actors)

        self.gamma = args.gamma

        # training config
        self.n_episodes = args.n_episodes
        self.episode_max_length = args.episode_max_length
        self.actor_optimizers = [Adam(actor.parameters(), lr=args.lr, weight_decay=0.01) for actor in self.actors]
        self.critic_optimizer = Adam(self.critic.parameters(), lr=args.lr, weight_decay=0.01)

        # replay_memory
        self.replay_memory = ConditionReplayMemory(args.max_memory_size, device=device)

        # # save checkpoints
        # self.model_dir = args.model_dir
        # if not os.path.exists(self.model_dir):
        #     os.makedirs(self.model_dir)
        # self.save_interval = args.save_interval
        #
        # # log
        self.k = 500
        self.writer = SummaryWriter(args.log_dir)
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)

    def calculate_returns(self, last_state_value, rewards):
        returns = []
        for rew in rewards[::-1]:
            last_state_value = self.gamma * last_state_value + rew
            returns = [last_state_value] + returns
        return torch.cat(returns)

    def learn(self):
        for actor in self.actors:
            actor.train()
        self.critic.train()

        # collect experience
        print("collecting experience")
        for _ in range(self.args.episodes_before_training):
            # rollout
            state = self.env.reset()
            state_per_step = []
            action_per_step = []
            reward_per_step = []
            for step in range(1, self.episode_max_length + 1):
                state = process_state(state)
                state = torch.tensor([state], device=self.device)
                state_per_step.append(state)
                actions = np.random.randint(0, len(self.action_list), size=self.n_players)

                obs, rew, done, info = self.env.step([self.action_list[action] for action in actions])
                action_per_step.append(torch.tensor(actions, device=self.device))
                rew = rew[0]
                reward_per_step.append(torch.tensor([rew], device=self.device))
                state = obs
                # 如果变成对方持球，就强行停止
                if self.args.early_stop:
                    if state[0, 96] == 1:
                        rew = -1
                        break
                if done:
                    break

            # add to memory

            returns = self.calculate_returns(last_state_value=0,
                                             rewards=reward_per_step)
            self.replay_memory.add_experience(states=torch.cat(state_per_step),
                                              actions=torch.cat(action_per_step).view(-1, self.n_players),
                                              returns=returns,
                                              reward_of_episode=rew
                                              )
        # training
        print("training")
        reward_per_episode = []
        reward_per_episode_without_early_stop = []
        step_per_episode = []
        entropy_per_episode = []
        critic_loss_per_episode = []
        for episode in range(1, self.n_episodes + 1):
            early_stop = False
            # rollout
            state = self.env.reset()
            state_per_step = []
            action_per_step = []
            reward_per_step = []
            for step in range(1, self.episode_max_length + 1):
                state = process_state(state)
                state = torch.tensor([state], device=self.device)
                state_per_step.append(state)
                actions_per_player = []
                for idx, actor in enumerate(self.actors):
                    dist = actor(state)
                    action = dist.sample()  # tensor(14)
                    action_per_step.append(action)
                    actions_per_player.append(self.action_list[action[0].item()])
                # for action in actions_per_player:
                #     print(football_action_set.named_action_from_action_set(self.env.unwrapped._env._action_set, action))
                obs, rew, done, info = self.env.step(actions_per_player)
                rew = rew[0]
                reward_per_step.append(torch.tensor([rew], device=self.device))
                state = obs
                # 如果变成对方持球，就强行停止
                if self.args.early_stop:
                    if state[0, 96] == 1:
                        rew = -1
                        early_stop = True
                        break
                if done:
                    break
            reward_per_episode.append(rew)
            if early_stop:
                reward_per_episode_without_early_stop.append(0)
            else:
                reward_per_episode_without_early_stop.append(rew)

            step_per_episode.append(step)

            # add to memory

            returns = self.calculate_returns(last_state_value=0,
                                             rewards=reward_per_step)
            self.replay_memory.add_experience(states=torch.cat(state_per_step),
                                              actions=torch.cat(action_per_step).view(-1, self.n_players),
                                              returns=returns,
                                              reward_of_episode=rew
                                              )

            # update
            batch_states, batch_actions, batch_returns = self.replay_memory.sample_minibatch(self.args.batch_size)

            values = self.critic(batch_states).squeeze(-1)

            advantages = batch_returns - values
            critic_loss = F.mse_loss(input=values, target=batch_returns.detach(), reduction="mean")
            critic_loss_per_episode.append(critic_loss.item())
            critic_loss.backward()
            self.critic_optimizer.step()
            self.critic_optimizer.zero_grad()

            entropy_per_player = []
            for idx, actor in enumerate(self.actors):
                dist = actor(batch_states)
                log_prob = dist.log_prob(batch_actions[:, idx])
                entropy = dist.entropy()
                entropy_per_player.append(entropy.mean().item())

                actor_loss = -log_prob * advantages.detach()
                actor_loss = actor_loss.mean()
                actor_loss.backward()
                self.actor_optimizers[idx].step()
                self.actor_optimizers[idx].zero_grad()
            entropy_per_episode.append(entropy_per_player)

            # log

            moving_average_window_size = len(reward_per_episode[-self.k:])
            moving_average_reward = sum(reward_per_episode[-self.k:]) / moving_average_window_size
            moving_average_reward_without_early_stop = sum(
                reward_per_episode_without_early_stop[-self.k:]) / moving_average_window_size

            moving_average_step = sum(step_per_episode[-self.k:]) / moving_average_window_size

            moving_average_entropy_per_player = [sum(e) / moving_average_window_size for e in
                                                 zip(*entropy_per_episode[-self.k:])]
            moving_average_critic_loss = sum(critic_loss_per_episode[-self.k:]) / moving_average_window_size
            print(
                f"episode: {episode}, reward: {reward_per_episode[-1]}, step: {step_per_episode[-1]}, moving average reward: {moving_average_reward}, without early stop: {moving_average_reward_without_early_stop}, moving average step: {moving_average_step}, moving average critic loss: {moving_average_critic_loss}, moving average entropy: {moving_average_entropy_per_player}")

            self.writer.add_scalar("moving average reward", moving_average_reward, episode)
            self.writer.add_scalar("moving average reward without early stop", moving_average_reward_without_early_stop,
                                   episode)

            self.writer.add_scalar("moving average step", moving_average_step, episode)
            self.writer.add_scalar("moving average critic loss", moving_average_critic_loss, episode)
            for idx in range(self.n_players):
                self.writer.add_scalar(f"moving average entropy of {idx}", moving_average_entropy_per_player[idx],
                                       episode)

