''' 
PPO discrete + CNN for single frame Breakout Atari
主函数实现，不包含功能测试/模块测试
'''

import gymnasium as gym
from ale_py import ALEInterface
from torch.utils.tensorboard import SummaryWriter
from ppo_breakout import PPO
from tqdm import tqdm
import cv2
import numpy as np
import os

def write_to_tensorboard(writer, episode, losses_data, perf_data):
    writer.add_scalar(tag='Perf/reward', scalar_value=perf_data['reward'], global_step=episode)
    writer.add_scalar(tag='Perf/success_rate', scalar_value=perf_data['success_rate'], global_step=episode)
    writer.add_scalar(tag='Perf/steps', scalar_value=perf_data['steps'], global_step=episode)
    writer.add_scalar(tag='Losses/actor_loss', scalar_value=losses_data['actor_loss'], global_step=episode)
    writer.add_scalar(tag='Losses/critic loss', scalar_value=losses_data['critic_loss'], global_step=episode)
    writer.add_scalar(tag='Losses/entropy', scalar_value=losses_data['entropy'], global_step=episode)

def preprocess_obs(obs, print_obs=False):
    obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
    obs = cv2.resize(obs, (84, 84))
    if print_obs:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(current_dir, "image.png")
        cv2.imwrite(save_path, obs) # (H, W, C)
    obs = obs / 255.0
    obs = np.expand_dims(obs,axis=0)  # (C, H, W)
    return obs.astype(np.float32)

if __name__ == "__main__":
    # initialize env
    env = gym.make("ALE/Breakout-v5", render_mode="human")
    action_space = env.action_space.n
    observation_space = env.observation_space.shape
    print("Action space:", action_space)    # 动作空间Discrete：（Noop, Fire, Left, Right）分别是（0, 1, 2, 3）
    print("Observation space:", observation_space)  # 状态空间Box（Height, Width, Channel）分别是（210, 160, 3）
    
    # initialize ppo
    episodes  = 1000
    write_data = True
    visualize_first_frame = True
    actor_lr = 1e-3
    critic_lr = 1e-3
    gamma = 0.99
    lmbda = 0.95
    output_state_dim = 256
    agent = PPO(output_state_dim, 128, action_space, actor_lr=0.001, critic_lr=0.005, gamma=0.99, lmbda=0.9, epsilon=0.3, epoch=10, image_size=84)

    if write_data == True:
        writer = SummaryWriter(log_dir=f'log/ppo_cnn')
        summary_window = 20
    
    with tqdm(total=episodes, desc="Training Episodes") as pbar:
        for epi in range(episodes):
            trajectory = {'state': [], 'action': [], 'reward': [], 'next_state': [], "done": []}
            state, info = env.reset()
            frame = preprocess_obs(state)
            done = False
            while not done:
                action = agent.take_action(frame)
                next_state, reward, terminated, truncated, info = env.step(action)
                next_frame = preprocess_obs(next_state)
                done = terminated or truncated
                trajectory['state'].append(frame)
                trajectory['action'].append(action)
                trajectory['reward'].append(reward)
                trajectory['done'].append(done)
                trajectory['next_state'].append(next_frame)
            agent.update(trajectory)
            pbar.update(1)