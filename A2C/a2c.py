# -*- coding: utf-8 -*-
# @Time        : 2020/4/19 18:22
# @Author      : ssxy00
# @File        : a2c.py
# @Description :

import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from tensorboardX import SummaryWriter


class A2C:
    def __init__(self, env, model, a2c_config, action_list):
        self.env = env
        self.action_list = action_list
        self.action_space_size = len(action_list)

        self.model = model

        self.gamma = a2c_config.gamma
        self.critic_coef = a2c_config.critic_coef
        self.entropy_coef = a2c_config.entropy_coef

        # training config

        self.device = torch.device(a2c_config.device) if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)
        self.max_episodes = a2c_config.max_episodes
        self.n_steps = a2c_config.n_steps
        self.optimizer = Adam(self.model.parameters(), lr=a2c_config.lr, weight_decay=0.01)
        # self.criterion = nn.MSELoss()

        # save checkpoints
        self.save_dir = a2c_config.save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.save_interval = a2c_config.save_interval

        # log
        self.k = a2c_config.k
        self.writer = SummaryWriter(a2c_config.log_dir)
        if not os.path.exists(a2c_config.log_dir):
            os.makedirs(a2c_config.log_dir)

    def calculate_returns(self, last_state_value, rewards):
        returns = []
        for rew in rewards[::-1]:
            last_state_value = self.gamma * last_state_value + rew
            returns = [last_state_value] + returns
        return torch.cat(returns)

    def learn(self):
        self.model.train()
        episode = 1

        episode_rewards = [] # 存储每一个 episode 的 reward
        episode_entropys = [] # 存储每一个 episode 的平均 entropy
        episode_steps = [] # 存储每一个 episode 的 step 数

        batch_loss = [] # 存储每一次 update 的 loss
        batch_actor_loss = [] # 存储每一次 update 的 actor loss
        batch_critic_loss = [] # 存储每一次 update 的 critic loss
        batch_entropys = [] # 存储每一次 update 的 entropy

        steps = 0 # 记录当前 episode 有多少个 step
        step_entropys = [] # 记录当前 episode 每个 step 的 entropy，用来算平均

        done = False
        update_nums = 0
        state = self.env.reset()

        while episode <= self.max_episodes:
            t = 0
            values = []
            rewards = []
            log_probs = []
            entropys = []

            # collecting experiments
            while t < self.n_steps and not done:
                state = torch.tensor(state, device=self.device)
                value, dist = self.model(state)
                action = dist.sample()
                log_prob = dist.log_prob(action).unsqueeze(0)
                entropy = dist.entropy().unsqueeze(0)

                obs, rew, done, info = self.env.step(self.action_list[action.item()])

                values.append(value)
                rewards.append(torch.tensor([rew], device=self.device))
                log_probs.append(log_prob)
                entropys.append(entropy)
                step_entropys.append(entropy)

                state = obs
                t += 1
                steps += 1

            values = torch.cat(values)
            log_probs = torch.cat(log_probs)
            entropys = torch.cat(entropys)

            # calculate last state value
            if done:
                last_state_value = 0 # 用于计算 returns

                # log reward
                episode_rewards.append(rew)
                moving_average_reward = sum(episode_rewards[-self.k:]) / len(episode_rewards[-self.k:])
                self.writer.add_scalar('moving average reward', moving_average_reward, episode)

                # log entropy
                episode_average_entropy = torch.cat(step_entropys).mean().item()
                episode_entropys.append(episode_average_entropy)
                moving_average_entropy = sum(episode_entropys[-self.k:]) / len(episode_entropys[-self.k:])
                self.writer.add_scalar('moving average entropy', moving_average_entropy, episode)

                # log step
                episode_steps.append(steps)
                moving_average_steps = sum(episode_steps[-self.k:]) / len(episode_steps[-self.k:])
                self.writer.add_scalar('moving average steps', moving_average_steps, episode)

                print(f"episode: {episode}, reward: {rew}, steps: {steps}, "
                      f"moving average reward: {moving_average_reward}, moving average entropy: {moving_average_entropy}")

                # save checkpoint
                if episode % self.save_interval == 0:
                    torch.save(self.model.state_dict(), os.path.join(self.save_dir, f"checkpoint{episode}.pt"))

                # reset episode
                steps = 0
                step_entropys = []
                state = self.env.reset()
                done = False
                episode += 1

                # # 记录 shoot action 的 log_prob
                # value, dist = self.model(torch.tensor(state, device=self.device))
                # self.writer.add_scalar('shoot action lprob',
                #                        dist.log_prob(torch.tensor(1, dtype=torch.long, device=value.device)).item(), episode)

            else:
                last_state_value, _ = self.model(torch.tensor(state, device=self.device))

            # bootstrapping returns
            returns = self.calculate_returns(last_state_value=last_state_value,
                                             rewards=rewards)

            # update
            advantages = returns - values
            critic_loss = F.mse_loss(input=values, target=returns.detach(), reduction="mean")
            actor_loss = -log_probs * advantages.detach()
            actor_loss = actor_loss.mean()
            policy_entropy = entropys.mean()

            loss = actor_loss + self.critic_coef * critic_loss - self.entropy_coef * policy_entropy

            # moving average mean
            batch_loss.append(loss.item())
            batch_actor_loss.append(actor_loss.item())
            batch_critic_loss.append(critic_loss.item())
            batch_entropys.append(policy_entropy.item())

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            update_nums += 1

            # loss log
            # time step
            self.writer.add_scalar('loss', loss.item(), update_nums)
            self.writer.add_scalar('actor_loss', actor_loss.item(), update_nums)
            self.writer.add_scalar('critic_loss', critic_loss.item(), update_nums)
            self.writer.add_scalar('policy_entropy', policy_entropy.item(), update_nums)

            # moving average mean
            window_size = len(batch_loss[-self.k:]) # 因为刚开始取不到 k 个
            self.writer.add_scalar('moving_average_loss', sum(batch_loss[-self.k:]) / window_size, update_nums)
            self.writer.add_scalar('moving_average_actor_loss', sum(batch_actor_loss[-self.k:]) / window_size, update_nums)
            self.writer.add_scalar('moving_average_critic_loss', sum(batch_critic_loss[-self.k:]) / window_size, update_nums)
            self.writer.add_scalar('moving_average_policy_entropy', sum(batch_entropys[-self.k:]) / window_size, update_nums)



