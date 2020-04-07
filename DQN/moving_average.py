# -*- coding: utf-8 -*-
# @Time        : 2020/4/7 20:05
# @Author      : ssxy00
# @File        : moving_average.py
# @Description : 试验中保存了每个 episode 的信息，和所有 episode 的平均信息，这里计算 reward, loss, steps 的滑动平均

from tensorboard.backend.event_processing import event_accumulator
from tensorboardX import SummaryWriter

def computing_moving_average(values, k):
    return_values = []
    for i in range(len(values)):
        start_index = max(0, i + 1 - k)
        end_index = min(len(values), i + 1)
        return_values.append(sum(values[start_index: end_index]) / (end_index - start_index))
    return return_values

def logging_moving_average(in_log_path, out_log_dir, k):
    # 加载日志数据
    ea = event_accumulator.EventAccumulator(in_log_path)
    ea.Reload()
    writer = SummaryWriter(out_log_dir)
    # reward
    val_reward = ea.scalars.Items('Train/reward')
    rewards = [item.value for item in val_reward]
    last_k_rewards = computing_moving_average(rewards, k)
    for idx, value in enumerate(last_k_rewards, 1):
        writer.add_scalar('Train/moving_average_reward', value, idx)
    # loss
    val_loss = ea.scalars.Items('Train/loss')
    loss = [item.value for item in val_loss]
    last_k_loss = computing_moving_average(loss, k)
    for idx, value in enumerate(last_k_loss, 1):
        writer.add_scalar('Train/moving_average_loss', value, idx)
    # steps
    val_steps = ea.scalars.Items('Train/steps')
    steps = [item.value for item in val_steps]
    last_k_steps = computing_moving_average(steps, k)
    for idx, value in enumerate(last_k_steps, 1):
        writer.add_scalar('Train/moving_average_steps', value, idx)

if __name__ == "__main__":
    in_log_path = "/path/to/training/log"
    out_log_dir = "/path/to/output/moving_average_logs"
    logging_moving_average(in_log_path, out_log_dir, k=500)

