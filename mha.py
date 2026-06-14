import torch
import gymnasium as gym
from ale_py import ALEInterface
from network import PolicyNet, CriticNet, FrameFeatureExtractor
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from collections import deque
import cv2
''' 在实现mha和CNN的基础上, 还需要做以下工作:
0. 明确CNN相当于输入图像的初步处理, mha只相当于一个特征提取器
1. 输入图像信息处理, 使用MHA_agent作为主逻辑所在类
'''
class MHA_Agent:
    def __init__(self, feature_dim, hidden_dim, action_dim, model_dim, cnn_out_dim, n_head, batch_size, seq_len, actor_lr, critic_lr, gamma, lmbda, epsilon, epoch=10, entropy_coef=0.01, device=None):
        self.critic_net = CriticNet(feature_dim, hidden_dim)
        self.actor_net = PolicyNet(feature_dim, hidden_dim, action_dim)
        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic_net.parameters(), lr=critic_lr)
        self.seq_len = seq_len
        self.frame_buffer = deque(maxlen=seq_len)   # 帧缓存，连续帧的信息从这里提取
        self.frameFeatureExtractor = FrameFeatureExtractor(cnn_out_dim, model_dim, n_head, batch_size, seq_len)  # cnn_out_dim表示cnn输出通道数目
        self.gamma = gamma
        self.lmbda = lmbda
        self.entropy_coef = entropy_coef
        self.epoch = epoch
        self.epsilon = epsilon
    def take_action(self, state):
        # 此时的state已经包含多帧
        frames = self.stack_frame(state)
        feature = self.extract_features(frames) # 包含时序信息的feature
        action_probs = self.actor_net(feature)
        action_dist = torch.distributions.Categorical(probs = action_probs)
        action = action_dist.sample()
        return action.item(), action_dist.log_prob(action).item()
    def update(self, trajectory):
        states = torch.tensor(trajectory['state'], dtype=torch.float)
        next_states = torch.tensor(trajectory['next_state'], dtype=torch.float)
        rewards = torch.tensor(trajectory['reward'], dtype=torch.float)
        actions = torch.tensor(trajectory['action'], dtype=torch.int64)
        dones = torch.tensor(trajectory['dones'], dtype=torch.int64)
        frames = self.stack_frame(states)
        feature = self.extract_features(frames)


        # 保留旧策略概率，用于比较
        with torch.no_grad():
            old_action_probs = self.actor_net(feature)
            old_action_dist = torch.distributions.Categorical(action_probs=old_action_probs)    # [steps] -> [steps, 1]
            old_action_logits = old_action_dist.log_prob(actions)
        
        for _ in range(self.epoch):
            value = self.critic_net(next_states).squeeze(-1) # [steps, 1]
            value_now = self.critic_net(states).squeeze(-1)
            TD_targets = rewards + self.gamma * value
            TD_deltas = TD_targets - value_now
            advantages = torch.zeros_like(TD_deltas)
            for i in reversed(range(len(trajectory['reward']))):
                if i < len(trajectory['reward']) - 1:
                    advantages[i] = advantages[i + 1] * (self.gamma * self.lmbda) * (1 - dones[i])
                advantages[i] += TD_deltas[i]
            advantages = advantages.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            action_dist = torch.distributions.Categorical(action_probs = self.actor_net(feature))
            entropy = action_dist.entropy().mean()
            ratio = torch.exp(action_dist.log_prob(actions) - old_action_logits)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            actor_loss = -torch.mean(torch.min(surr1, surr2)) - self.entropy_coef * entropy
            critic_loss = torch.mean(F.mse_loss(value_now, TD_targets.detach()))

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
        losses_data = {'actor_loss': actor_loss, 'critic_loss': critic_loss}
        perf_data = {'reward': np.sum(trajectory['reward']), 'steps': len(trajectory['reward'])}
        return losses_data, perf_data
    # 预处理step 0: 原始图象标准化
    def preprocess_frame(self, frame):
        if len(frame.shape) == 3:
            print("图象为RGB图象，转为灰度图")
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
        frame = frame.astype(np.float32) / 255.0    # 归一化
        return frame
    
    # 预处理step 1: 标准化后图象进行堆叠
    def stack_frame(self, frames):
        if frames is not None:
            # several frames
            for fr in frames:
                processed_frame = self.preprocess_frame(fr)
                self.frame_buffer.append(processed_frame)
        while len(self.frame_buffer) < self.seq_len:
            if len(self.frame_buffer) == 0:
                self.frame_buffer.append(np.zeros(84, 84, dtype=np.float32))
            else:
                self.frame_buffer.append(self.frame_buffer[-1]) # 用最新一帧代替
        sequence = np.stack(list(self.frame_buffer), axis=0)
        return sequence

    # 预处理step 2: 堆叠后图象序列通过FrameExtractor提取特征
    def extract_features(self, frame_sequence):
        # shape -> (batch, seq_len, c, h, w)
        if not isinstance(frame_sequence, torch.Tensor):
            frame_sequence = torch.tensor(frame_sequence, dtype=torch.float32) 
        if frame_sequence.ndim == 3:
            frame_sequence.unsqueeze(0)
        if frame_sequence.ndim == 4:
            # batch == 1 or c == 1
            if frame_sequence.shape[0] == self.seq_len:
                frame_sequence.unsqueeze(0)
            else:
                frame_sequence.unsqueeze(2)
        features = self.frameFeatureExtractor(frame_sequence)
        return features
