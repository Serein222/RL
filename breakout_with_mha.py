''' 
PPO discrete + CNN + mha for multiple frames Breakout Atari
'''

import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
from ppo_discrete import PPO
from tqdm import tqdm

def write_to_tensorboard(writer, episode, losses_data, perf_data):
    writer.add_scalar(tag='Perf/reward', scalar_value=perf_data['reward'], global_step=episode)
    writer.add_scalar(tag='Perf/success_rate', scalar_value=perf_data['success_rate'], global_step=episode)
    writer.add_scalar(tag='Perf/steps', scalar_value=perf_data['steps'], global_step=episode)
    writer.add_scalar(tag='Losses/actor_loss', scalar_value=losses_data['actor_loss'], global_step=episode)
    writer.add_scalar(tag='Losses/critic loss', scalar_value=losses_data['critic_loss'], global_step=episode)
    writer.add_scalar(tag='Losses/entropy', scalar_value=losses_data['entropy'], global_step=episode)

if __name__ == "__main__":
    # parameters
    episodes  = 1000
    write_data = True
    env = gym.make("ALE/Breakout-v5", render_mode="human")
    
    action_space = env.action_space.n
    observation_space = env.observation_space.shape
    print("Action space:", action_space)    # 动作空间Discrete：（Noop, Fire, Left, Right）分别是（0, 1, 2, 3）
    print("Observation space:", observation_space)  # 状态空间Box（Height, Width, Channel）分别是（210, 160, 3）
    visualize_first_frame = True
    actor_lr = 1e-3
    critic_lr = 1e-3
    gamma = 0.99
    lmbda = 0.95
    mha_agent = MHA_Agent(
                feature_dim=observation_space, 
                hidden_dim=128, 
                action_dim=action_space, 
                model_dim=128, 
                cnn_out_dim=64, 
                n_head=4, 
                batch_size=1, 
                seq_len=4, 
                actor_lr=actor_lr, 
                critic_lr=critic_lr, 
                gamma=gamma, 
                lmbda=lmbda, 
        epsilon=0.2)
    if write_data == True:
        writer = SummaryWriter(log_dir=f'log/mha')
        summary_window = 20
    state, info = env.reset()
    if visualize_first_frame:
        # 测试第一帧图像提取效果
        mha_agent.frameFeatureExtractor(state, visualize_first_frame)
        vissualize_feature = False
    with tqdm(total=episodes, desc="Training Episodes") as pbar:
        for epi in range(episodes):
            trajectory = {'state': [], 'action': [], 'reward': [], 'next_state': []}
            state, info = env.reset()
            done = False
            while not done:
                action = mha_agent.take_action(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                trajectory['state'].append(state)
                trajectory['action'].append(action)
                trajectory['reward'].append(reward)
                trajectory['dones'].append(done)
                trajectory['next_state'].append(next_state)
            mha_agent.update(trajectory)
            pbar.update(1)