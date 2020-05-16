# A2C(Multiagent)
用基于 A2C 的 multi-agent 算法实现 gfootball environment `academy_3_vs_1_with_keeper`

## 文件说明
### 训练部分
+ without replay buffer: \
`python train.py`
+ with replay buffer \
`python train_replay.py`

### 参数设置
命令行传入

### 神经网络结构
`model.py`

### A2C 算法实现
+ with replay buffer: \
`maa2c.py` 
+ without replay buffer: \
`maa2c_replay.py`