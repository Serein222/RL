      
import gymnasium as gym
import torch
import numpy as np
from .network import PolicyNet, CriticNet
import torch.nn.functional as F
import tensorboard

class ActorCritic:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, device=None):
        self.actor_net = PolicyNet(state_dim, hidden_dim, action_dim)
        self.critic_net = CriticNet(state_dim, hidden_dim)
        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic_net.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.device = device

    def take_action(self, state):
        action_probs = self.actor_net.forward(state)
        action_dist = torch.distributions.categorical(action_probs)
        action = action_dist.sample()
        return action.item()


    def update(self, trajectory):
        states = torch.tensor(trajectory['state'], dtype=torch.float32)
        next_states = torch.tensor(trajectory['next_state'], dtype=torch.float32)
        actions = torch.tensor(trajectory['action'], dtype=torch.int64)
        rewards = torch.tensor(trajectory['reward'], dtype=torch.float32)
        values_next = self.critic_net.forward(next_states)
        values_now = self.critic_net.forward(states)
        TD_targets = rewards + self.gamma * values_next
        TD_deltas = TD_targets - values_now

        action_probs = self.actor_net.forward(states)
        action_logits = torch.log(action_probs)
        actor_loss = -torch.mean(TD_deltas * torch.log(action_logits[actions]))
        critic_loss = torch.mean(F.mse_loss(TD_targets, values_now))
        # F.mse_loss(TD_targets, values_now)等价于TD_deltas^2
        
        # 在每个episode结束时更新
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

if __name__ == "__main__":
    env = gym.make("frozenlake-v1", is_slippery=False, render_mode="human")
    action_space = env.action_space.n
    observation_space = env.observation_space.n
    print("Action space:", action_space)
    print("Observation space:", observation_space)
    agent = ActorCritic(observation_space, 128, action_space, learning_rate=0.01, gamma=0.9)
    
    trajectory = {'state': [], 'action': [], 'reward': []}
    terminate = False
    truncated = False
    state, info = env.reset()
    done = terminate or truncated
    for episode in range(1000):
        while not done:  
            action = agent.take_action(state)
            state, reward, terminate, truncated, info = env.step(action)
            trajectory['state'].append(state)
            trajectory['action'].append(action)
            trajectory['reward'].append(reward)
            done = terminate or truncated
        
        agent.update(trajectory)
        trajectory = {'state': [], 'action': [], 'reward': []}
        env.reset()