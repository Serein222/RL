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
        raw_states = trajectory['state']
        # print(states)
        # print("debug: states type:", type(states))
        # print("debug: states shape:", states.shape)
        raw_next_states = trajectory['next_state']
        actions = torch.tensor(trajectory['action'], dtype=torch.int64)
        actions = actions.unsqueeze(1)  # (steps, 1)
        # if actions.ndim == 0:
        #     actions.unsqueeze(0)    # -> (1, bacth_size)
        # actions = actions.unsqueeze(-1) # (1, bacth_size, 1)
        rewards = torch.tensor(trajectory['reward'], dtype=torch.float32)
        dones = torch.tensor(trajectory['done'], dtype=torch.float32)

        # =========================
        # 1. 计算 old policy / advantage
        # =========================
        with torch.no_grad():
            states_feature = self.extractor(raw_states)
            next_states_feature = self.extractor(raw_next_states)
            action_probs = self.actor_net(states_feature)
            # print("debug: ", action_probs.ndim, action_probs.shape, actions, actions.ndim, actions.shape)
            old_action_logits = torch.log(action_probs.gather(1, actions))    # [steps] -> [steps, 1]
            values_next = self.critic_net(next_states_feature).squeeze(-1)
            values_now = self.critic_net(states_feature).squeeze(-1)
            TD_targets = rewards + self.gamma * values_next * (1 - dones)
            TD_deltas = TD_targets - values_now.detach()    # [steps]
            advantages = TD_deltas.clone()  # [steps]
            # print("debug: advantages shape:", advantages.shape)
            for i in reversed(range(len(TD_deltas))):
                if i < len(TD_deltas) - 1:
                    advantages[i] += advantages[i + 1] * (self.gamma * self.lmbda)
            
            advantages = advantages.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        total_steps = len(actions)
        # =========================
        # 2. PPO mini batch update
        # =========================

        for _ in range(self.epoch):
            indices = np.arange(total_steps)
            np.random.shuffle(indices)
            for start in range(0, total_steps, self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]
                batch_idx = torch.tensor(batch_idx, dtype=torch.long)
                batch_states = [raw_states[i] for i in batch_idx]
                batch_actions = actions[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_old_log_probs = old_action_logits[batch_idx]
                batch_targets = TD_targets[batch_idx]
                # =====================
                # 重新建立CNN计算图
                # =====================
                features = self.extractor(batch_states)
                action_probs = self.actor_net(features)
                action_dist = torch.distributions.Categorical(probs=action_probs)
                entropy = action_dist.entropy().mean()
                log_probs = torch.log(action_probs.gather(1, batch_actions))
                ratio = torch.exp(log_probs - batch_old_log_probs).squeeze(-1)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
                values = self.critic_net(features).squeeze(-1)
                critic_loss = F.mse_loss(values, batch_targets) # 默认已经求均值了
                loss = (actor_loss + 0.5 * critic_loss)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        losses_data = {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": entropy.item()
        }

        perf_data = {
            "reward": np.sum(trajectory['reward']),
            "steps": len(trajectory['reward'])
        }
        return losses_data, perf_data


    