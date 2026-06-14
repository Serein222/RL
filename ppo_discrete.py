import gymnasium as gym
import numpy as np
from network import PolicyNet, CriticNet
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch
from tqdm import tqdm

class PPO:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, lmbda, epsilon, epoch, entropy_coef=0.01, device=None):
        self.actor_net = PolicyNet(state_dim, hidden_dim, action_dim)
        self.critic_net = CriticNet(state_dim, hidden_dim)
        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic_net.parameters(), lr=critic_lr)
        self.state_dim = state_dim
        self.gamma = gamma
        self.lmbda = lmbda
        self.entropy_coef = entropy_coef
        self.epsilon = epsilon
        self.epoch = epoch
        self.device = device

    def take_action(self, state):
        action_probs = self.actor_net(self.one_hot(state, self.state_dim))
        action_dist = torch.distributions.Categorical(probs=action_probs)
        action = action_dist.sample()
        return action.item(), action_dist.log_prob(action).item()
    
    def one_hot(self, data_list, dim):
        if not isinstance(data_list, list):
            data_list = [data_list]
        num = len(data_list)
        data_onehot = torch.zeros((num, dim), dtype=torch.float32)
        data_index = torch.tensor(data_list, dtype=torch.int64)
        data_onehot[torch.arange(num), data_index] = 1.0
        return data_onehot
    
    def update(self, trajectory):
        states = self.one_hot(trajectory['state'], self.state_dim)
        next_states = self.one_hot(trajectory['next_state'], self.state_dim)
        actions = torch.tensor(trajectory['action'], dtype=torch.int64)
        rewards = torch.tensor(trajectory['reward'], dtype=torch.float32)
        success = 0
        if torch.max(rewards) == 50:
            success = 1
        with torch.no_grad():
            action_probs = self.actor_net(states)
            old_action_logits = torch.log(action_probs.gather(1, actions.unsqueeze(-1)))    # [steps] -> [steps, 1]
        
        for epoch_idx in range(self.epoch):
            with torch.no_grad():
                values_next = self.critic_net(next_states).squeeze(-1)
                TD_targets = rewards + self.gamma * values_next
            values_now = self.critic_net(states).squeeze(-1)
            TD_deltas = TD_targets - values_now.detach()    # [steps]

            action_probs = self.actor_net(states)
            action_dist = torch.distributions.Categorical(probs=action_probs)   # [steps, action_dim]
            entropy = action_dist.entropy().mean()  # [steps] -> [1]
            action_logits = torch.log(action_probs.gather(1, actions.unsqueeze(-1))) # [steps, 1]
            ratio = torch.exp(action_logits - old_action_logits).squeeze(-1)
            advantages = TD_deltas.clone()
            for i in reversed(range(len(TD_deltas))):
                if i < len(TD_deltas) - 1:
                    advantages[i] += advantages[i + 1] * (self.gamma * self.lmbda)
            
            advantages = advantages.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            actor_loss = -torch.mean(torch.min(surr1, surr2)) - self.entropy_coef * entropy
            critic_loss = torch.mean(F.mse_loss(TD_targets.detach(), values_now))   # 最小化预测误差不要加负号

            # 在每个epoch结束时更新
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
        
        losses_data = {'actor_loss': actor_loss.item(), 'critic_loss': critic_loss.item(), 'entropy': entropy.item()}
        perf_data = {'reward': np.sum(trajectory['reward']), 'success_rate': success, 'steps': len(trajectory['reward'])}
        return losses_data, perf_data


    