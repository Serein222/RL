import numpy as np
from network import PolicyNet, CriticNet
from cnn import FrameFeatureExtractor
import torch.nn.functional as F
import torch
from collections import deque
class PPO:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, lmbda, epsilon, epoch, image_size, stuck_frame=None, batch_size=64, entropy_coef=0.001, device=None):
        self.extractor = FrameFeatureExtractor(image_size, state_dim, stuck_frame)  # 传入堆叠的帧数
        self.actor_net = PolicyNet(state_dim, hidden_dim, action_dim)
        self.critic_net = CriticNet(state_dim, hidden_dim)
        # self.actor_optimizer = torch.optim.Adam(list(self.actor_net.parameters()) + list(self.extractor.parameters()), lr=actor_lr)
        # self.critic_optimizer = torch.optim.Adam(list(self.critic_net.parameters()) + list(self.extractor.parameters()), lr=critic_lr)
        self.optimizer = torch.optim.Adam(
            list(self.extractor.parameters())
            +
            list(self.actor_net.parameters())
            +
            list(self.critic_net.parameters()),
            critic_lr
        )
        self.state_dim = state_dim
        self.gamma = gamma
        self.lmbda = lmbda
        self.entropy_coef = entropy_coef
        self.epsilon = epsilon
        self.epoch = epoch
        self.device = device
        self.stuck_frame = stuck_frame
        self.batch_size = batch_size
    def take_action(self, state):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state) # -> (C, H, W)
        if isinstance(state, deque) and self.stuck_frame is not None:
            frames = np.array(state, dtype=np.float32)  # state is (H, W)
            state = torch.from_numpy(frames) # -> (frame, H, W)
        action_probs = self.actor_net(self.extractor(state)[0])
        action_dist = torch.distributions.Categorical(probs=action_probs)
        action = action_dist.sample()
        return action.item(), action_dist.log_prob(action).item()
    
    def update(self, trajectory):
        states = self.extractor(trajectory['state'])
        # print(states)
        # print("debug: states type:", type(states))
        # print("debug: states shape:", states.shape)
        next_states = self.extractor(trajectory['next_state'])
        actions = torch.tensor(trajectory['action'], dtype=torch.int64)
        actions = actions.unsqueeze(1)  # (steps, 1)
        # if actions.ndim == 0:
        #     actions.unsqueeze(0)    # -> (1, bacth_size)
        # actions = actions.unsqueeze(-1) # (1, bacth_size, 1)
        rewards = torch.tensor(trajectory['reward'], dtype=torch.float32)
        dones = torch.tensor(trajectory['done'], dtype=torch.int64)
        with torch.no_grad():
            # 计算old log probs / TD target / Advantages
            action_probs = self.actor_net(states)
            # print("debug: ", action_probs.ndim, action_probs.shape, actions, actions.ndim, actions.shape)
            old_action_logits = torch.log(action_probs.gather(1, actions))    # [steps] -> [steps, 1]
            values_next = self.critic_net(next_states).squeeze(-1)
            values_now = self.critic_net(states).squeeze(-1)
            TD_targets = rewards + self.gamma * values_next * (1 - dones)
            TD_deltas = TD_targets - values_now.detach()    # [steps]
            advantages = TD_deltas.clone()  # [steps]
            # print("debug: advantages shape:", advantages.shape)
            for i in reversed(range(len(TD_deltas))):
                if i < len(TD_deltas) - 1:
                    advantages[i] += advantages[i + 1] * (self.gamma * self.lmbda)
            
            advantages = advantages.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(self.epoch):
            states = self.extractor(trajectory['state'])
            action_probs = self.actor_net(states)
            action_dist = torch.distributions.Categorical(probs=action_probs)   # [steps, action_dim]
            entropy = action_dist.entropy().mean()  # [steps] -> [1]
            
            action_logits = torch.log(action_probs.gather(1, actions)) # [steps, 1]
            ratio = torch.exp(action_logits - old_action_logits).squeeze(-1)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            actor_loss = -torch.mean(torch.min(surr1, surr2)) - self.entropy_coef * entropy
            values_now = self.critic_net(states).squeeze(-1)    # 预测当前的values, 目标是固定的TD_targets
            critic_loss = torch.mean(F.mse_loss(TD_targets.detach(), values_now))   # 最小化预测误差不要加负号

            # 在每个epoch结束时更新
            # self.actor_optimizer.zero_grad()
            # self.critic_optimizer.zero_grad()
            # actor_loss.backward(retain_graph=True)
            # critic_loss.backward(retain_graph=True)
            # self.actor_optimizer.step()
            # self.critic_optimizer.step()
            self.optimizer.zero_grad()
            loss = actor_loss + critic_loss * 0.5
            loss.backward()
            self.optimizer.step()
        losses_data = {'actor_loss': actor_loss.item(), 'critic_loss': critic_loss.item(), 'entropy': entropy.item()}
        perf_data = {'reward': np.sum(trajectory['reward']), 'steps': len(trajectory['reward'])}
        return losses_data, perf_data


    