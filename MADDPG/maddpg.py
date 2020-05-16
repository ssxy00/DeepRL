# -*- coding: utf-8 -*-
# @Time        : 2020/5/12 22:00
# @Author      : ssxy00
# @File        : maddpg.py
# @Description :

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

from utils import ReplayMemory, soft_update, convert_to_onehot, gumbel_softmax


class MADDPG:
    def __init__(self, env, action_list, actors, critics, old_actors, old_critics, args, device):
        self.device = device
        self.env = env
        self.n_players = len(actors)
        self.action_list = action_list
        self.action_space_size = len(action_list)
        self.actors = [actor.to(device) for actor in actors]
        self.critics = [critic.to(device) for critic in critics]
        self.old_actors = [old_actor.to(device) for old_actor in old_actors]
        self.old_critics = [old_critic.to(device) for old_critic in old_critics]
        self.old_actors = old_actors
        self.old_critics = old_critics
        # self.max_memory_size = args.max_memory_size
        self.replay_memory = ReplayMemory(max_memory_size=args.max_memory_size)
        self.episodes_before_training = args.episodes_before_training
        self.n_episodes = args.n_episodes
        self.episode_max_length = args.episode_max_length
        self.batch_size = args.batch_size
        self.save_interval = args.save_interval

        self.gamma = args.gamma
        self.epsilon = args.epsilon
        self.tau = args.tau

        self.critic_loss = nn.MSELoss()
        self.lr = args.lr
        self.actor_optimizers = [Adam(model.parameters(), lr=args.lr, weight_decay=0.01) for model in self.actors]
        self.critic_optimizers = [Adam(model.parameters(), lr=args.lr, weight_decay=0.01) for model in self.critics]

        # save checkpoints
        # self.model_dir = args.model_dir
        # if not os.path.exists(self.model_dir):
        #     os.makedirs(self.model_dir)
        # self.save_interval = args.save_interval

        # log
        self.k = 500  # moving average window size
        self.writer = SummaryWriter(args.log_dir)
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)

    @staticmethod
    def process_state(state):
        # 将 env 返回的每个 agent 的 state 整合成一个 state
        processed_state = state[0]
        processed_state[98:101] = 1
        return processed_state

    def actor_select_action(self, actor, state):
        action = actor(torch.tensor([state], device=self.device)).squeeze(0).argmax(-1).item()
        return action

    def select_actions(self, state):
        actions = [self.actor_select_action(actor, state) for actor in self.actors]
        return np.array(actions)

    def update(self):
        batch_states, batch_actions, batch_rewards, batch_new_states, batch_dones = self.replay_memory.sample_mini_batch(
            batch_size=self.batch_size)
        batch_states = batch_states.to(self.device)
        batch_actions = batch_actions.to(self.device)
        batch_rewards = batch_rewards.to(self.device)
        batch_new_states = batch_new_states.to(self.device)
        batch_dones = batch_dones.to(self.device)
        critic_loss_per_agent = []
        actor_loss_per_agent = []
        for idx in range(len(self.actors)):
            actor = self.actors[idx]
            critic = self.critics[idx]
            old_actor = self.old_actors[idx]
            old_critic = self.old_critics[idx]
            actor_optimizer = self.actor_optimizers[idx]
            critic_optimizer = self.critic_optimizers[idx]

            # update critic
            predict_Q = critic(state=batch_states, actions=batch_actions).squeeze(-1)
            old_actor_actions = old_actor(batch_new_states)

            target_actions = batch_actions.clone().detach()
            target_actions[:, idx, :] = old_actor_actions
            target_actions = convert_to_onehot(target_actions, epsilon=self.epsilon)
            target_Q = self.gamma * old_critic(state=batch_new_states, actions=target_actions).squeeze(
                -1) * (1 - batch_dones) + batch_rewards
            c_loss = self.critic_loss(input=predict_Q, target=target_Q.detach())
            c_loss.backward()
            torch.nn.utils.clip_grad_norm(critic.parameters(), 0.5)
            critic_optimizer.step()
            critic_optimizer.zero_grad()
            critic_loss_per_agent.append(c_loss.item())

            # update actor
            actor_actions = actor(batch_states)
            actor_actions = gumbel_softmax(actor_actions, hard=True)
            predict_actions = batch_actions.clone().detach()
            predict_actions[:, idx, :] = actor_actions
            a_loss = -critic(state=batch_states, actions=predict_actions).squeeze(-1)
            a_loss = a_loss.mean()
            torch.nn.utils.clip_grad_norm(actor.parameters(), 0.5)
            a_loss.backward()
            actor_optimizer.step()
            actor_optimizer.zero_grad()
            actor_loss_per_agent.append(a_loss.item())
        return sum(actor_loss_per_agent) / len(actor_loss_per_agent), sum(critic_loss_per_agent) / len(
            critic_loss_per_agent)

    def update_old_models(self):
        for old_model, model in zip(self.old_actors, self.actors):
            soft_update(old_model=old_model, current_model=model, tau=self.tau)
        for old_model, model in zip(self.old_critics, self.critics):
            soft_update(old_model=old_model, current_model=model, tau=self.tau)

    def learn(self):
        # before training
        episode_steps = []
        for _ in range(self.episodes_before_training):
            state = self.env.reset()
            state = self.process_state(state)
            for step in range(self.episode_max_length):
                actions = np.random.choice(self.action_list, size=self.n_players)
                obs, rew, done, info = self.env.step(actions)
                obs = self.process_state(obs)
                # check reward
                assert rew[0] == rew[1]
                assert rew[0] == rew[2]
                rew = rew[0]

                self.replay_memory.add_experience(state=state, actions=np.eye(self.action_space_size)[actions],
                                                  reward=rew, new_state=obs, done=done)
                state = obs
                if done:
                    episode_steps.append(step + 1)
                    break
        print(f"average steps: {sum(episode_steps) / len(episode_steps)}")

        # learn
        reward_per_episodes = []
        actor_loss_per_episodes = []
        critic_loss_per_episodes = []
        for episode in range(1, self.n_episodes + 1):
            state = self.env.reset()
            state = self.process_state(state)
            for step in range(1, self.episode_max_length + 1):
                # rollout
                actions = self.select_actions(state)
                obs, rew, done, info = self.env.step(actions)
                obs = self.process_state(obs)
                rew = rew[0]
                self.replay_memory.add_experience(state=state, actions=np.eye(self.action_space_size)[actions],
                                                  reward=rew, new_state=obs, done=done)
                # update
                a_loss, c_loss = self.update()

                # update old model
                self.update_old_models()

                state = obs
                if done:
                    break
            reward_per_episodes.append(rew)
            # TODO 这里实际上只记录到了每个 episode 结束那个 step 更新时的 loss，不全
            actor_loss_per_episodes.append(a_loss)
            critic_loss_per_episodes.append(c_loss)
            window_size = len(reward_per_episodes[-self.k:])
            moving_average_reward = sum(reward_per_episodes[-self.k:]) / window_size
            moving_average_actor_loss = sum(actor_loss_per_episodes[-self.k:]) / window_size
            moving_average_critic_loss = sum(critic_loss_per_episodes[-self.k:]) / window_size
            print(
                f"episode: {episode}, steps: {step}, reward: {rew}, moving average reward: {moving_average_reward}, actor loss: {a_loss}, critic loss: {c_loss}")
            self.writer.add_scalar('moving average reward', moving_average_reward, episode)
            self.writer.add_scalar('moving average actor loss', moving_average_actor_loss, episode)
            self.writer.add_scalar('moving average critic loss', moving_average_critic_loss, episode)


            # if episode % self.save_interval == 0:
            #     for idx, actor in enumerate(self.actors):
            #         torch.save(actor.state_dict(), os.path.join(self.model_dir, f"checkpoint{episode}_actor_{idx}.pt"))
            #     for idx, critic in enumerate(self.critics):
            #         torch.save(critic.state_dict(), os.path.join(self.model_dir, f"checkpoint{episode}_critic_{idx}.pt"))
