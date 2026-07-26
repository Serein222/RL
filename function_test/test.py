import gymnasium as gym
from ale_py import ALEInterface
import numpy as np
import cv2
import os
import sys
import torch
from tqdm import tqdm
from collections import deque
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ppo_breakout import PPO
from torch.utils.tensorboard import SummaryWriter

'''
函数功能测试/模块测试
'''
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
'''
添加模型检查
'''
def save_checkpoint(agent, global_step, save_dir="checkpoints", optimizer=None):
    """保存模型检查点，包含网络权重和优化器状态（可选）"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    checkpoint = {
        'global_step': global_step,
        'extractor_state_dict': agent.extractor.state_dict(),
        'actor_state_dict': agent.actor_net.state_dict(),
        'critic_state_dict': agent.critic_net.state_dict(),
    }
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    # 文件名可包含步数
    file_path = os.path.join(save_dir, f"checkpoint_step_{global_step}.pt")
    torch.save(checkpoint, file_path)
    print(f"Checkpoint saved to {file_path}")


'''
添加可视化
'''
def write_to_tensorboard(writer, episode, global_step, losses_data, perf_data):
    writer.add_scalar(tag='Perf/reward', scalar_value=perf_data['reward'], global_step=global_step)
    # writer.add_scalar(tag='Perf/success_rate', scalar_value=perf_data['success_rate'], global_step=episode)
    writer.add_scalar(tag='Perf/steps', scalar_value=perf_data['steps'], global_step=global_step)
    writer.add_scalar(tag='Losses/actor_loss', scalar_value=losses_data['actor_loss'], global_step=global_step)
    writer.add_scalar(tag='Losses/critic loss', scalar_value=losses_data['critic_loss'], global_step=global_step)
    writer.add_scalar(tag='Losses/entropy', scalar_value=losses_data['entropy'], global_step=global_step)

if __name__ == "__main__":
    env = gym.make("ALE/Breakout-v5", frameskip=4)   # , render_mode="human"
    writer = SummaryWriter(log_dir='log/stuck_frame')
    summary_window = 2048 * 5
    action_space = env.action_space.n
    observation_space = env.observation_space.shape
    print("Action space:", action_space)    # 动作空间Discrete：（Noop, Fire, Left, Right）分别是（0, 1, 2, 3）
    print("Observation space:", observation_space)  # 状态空间Box（Height, Width, Channel）分别是（210, 160, 3）
    print("has Frame skip:", env.unwrapped._frameskip)
    # initialize ppo
    episodes  = 10000
    rollout = 2048
    global_step = 0
    write_data = True
    visualize_first_frame = True
    actor_lr = 2.5e-4
    critic_lr = 5e-4
    gamma = 0.99
    lmbda = 0.95
    stuck_frame = 4 # n帧图像堆叠
    # state_dim取决于extractor返回的(batch, dim)的dim
    agent = PPO(256, 128, action_space, actor_lr, critic_lr, gamma, lmbda, epsilon=0.2, epoch=10, image_size=84, stuck_frame=4)
    trajectory = {'state': [], 'action': [], 'reward': [], 'next_state': [], "done": []}
    with tqdm(total=episodes, desc="Training Episodes") as pbar:
        for episode in range(episodes):
            state, info = env.reset()
            last_lives = info['lives']
            done = False
            action = -1
            frame_buffer = deque(maxlen=stuck_frame)
            while not done:
                # visualize preprocess
                image = preprocess_obs(state, True)
                # print(type(image), image.shape)
                if not frame_buffer:
                    # 用第一帧填充
                    for _ in range(stuck_frame):
                        frame_buffer.append(image[0])
                # visualize extractor
                # extractor = agent.extractor
                # out, x3, x2, x1 = extractor(image)
                # extractor.cnn.visualize_one_image(x1, "x1")
                # extractor.cnn.visualize_one_image(x2, "x2")
                # extractor.cnn.visualize_one_image(x3, "x3")

                # ppo take_action test
                # if action == -1:
                #     action = 1
                # else:
                action, prob = agent.take_action(frame_buffer)
                # print("output action is:", action, "its prob is:", np.exp(prob))

                # ppo step test
                next_frame, reward, terminated, truncated, info = env.step(action)
                next_image = preprocess_obs(next_frame)
                done = terminated or truncated
                lives = info['lives']
                if lives < last_lives:
                    reward -= 0.1   # 漏球惩罚
                # 如果要计算是否接到球，则需要屏蔽球尚未下落时板子的动作
                last_lives = lives
                # print(f"debug: info is {info}")
                # ppo update test
                trajectory['state'].append(frame_buffer)
                trajectory['action'].append(action)
                trajectory['reward'].append(reward)
                trajectory['done'].append(done)
                
                frame_buffer.append(next_image[0])
                trajectory['next_state'].append(frame_buffer)
                global_step += 1
                if write_data == True and global_step % summary_window == 0:
                    write_to_tensorboard(writer, episode, global_step, loss, perf)
                    # 模型保存
                    print(f"save model at {global_step} steps")
                    save_checkpoint(agent, global_step, save_dir='./model', optimizer=agent.optimizer)
                if len(trajectory['state']) >= rollout: # trajectory to rollout
                    loss, perf = agent.update(trajectory)   # 用完之后清空收集的trajectory
                    trajectory = {'state': [], 'action': [], 'reward': [], 'next_state': [], "done": []}
            # pbar.set_postfix({
            #     'Reward': f"{perf['reward']:.2f}",
            #     # 'Success Rate': f"{perf['success_rate']:.2f}",
            #     'Steps': perf['steps']
            # })
            pbar.update(1)
            
