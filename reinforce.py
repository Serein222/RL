      
import gymnasium as gym
import numpy as np
from .network import PolicyNet
import tensorboard
import torch

class Reinforce:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma, device=None):
        self.net = PolicyNet(state_dim, hidden_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        self.learning_rate = learning_rate
        self.gamma = gamma

    def take_action(self, state):
        action_prob = self.net.forward(state)
        action_dist = torch.distributions.categorical(action_prob)
        action = action_dist.sample()
        return action.item()
    
    def update(self, trajectory):
        actions = trajectory['action']
        states = trajectory['state']
        rewards = trajectory['reward']
        Gt = 0
        self.optimizer.zero_grad()
        for i, reward in enumerate(reversed(rewards)):
            Gt += Gt * self.gamma + reward
            log_probs = torch.log(self.forward(states[i]))
            loss = -Gt * log_probs[actions[i]]
            loss.backward()
        self.optimizer.step()



if __name__ == "__main__":
    env = gym.make("frozenlake-v1", is_slippery=False, render_mode="human")
    action_space = env.action_space.n
    observation_space = env.observation_space.n
    print("Action space:", action_space)
    print("Observation space:", observation_space)
    agent = Reinforce(observation_space, 128, action_space, learning_rate=0.01, gamma=0.9)
    net = agent.net
    trajectory = {'state': [], 'action': [], 'reward': []}
    state, info = env.reset()

    for episode in range(1000):
        action = net.take_action(state)
        state, reward, terminate, truncated, info = env.step(action)
        trajectory['state'].append(state)
        trajectory['action'].append(action)
        trajectory['reward'].append(reward)
        if terminate or truncated:
            net.update(trajectory)
            trajectory = {'state': [], 'action': [], 'reward': []}
            env.reset()
    
    env.close()




    