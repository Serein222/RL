''' 
PPO discrete for frozen lake
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
    write_data = True
    if write_data == True:
        writer = SummaryWriter(f'log/ppo_discrete')
        summary_window = 20 # record per 20 episodes
    else:
        writer = None
    env = gym.make("FrozenLake-v1", is_slippery=False, max_episode_steps=50)    # render_mode="human", 
    action_space = env.action_space.n
    observation_space = env.observation_space.n
    print("Action space:", action_space)    # 4
    print("Observation space:", observation_space)  # 16
    agent = PPO(observation_space, 128, action_space, actor_lr=0.001, critic_lr=0.005, gamma=0.99, lmbda=0.9, epsilon=0.3, epoch=10)
    
    trajectory = {'state': [], 'action': [], 'reward': [], 'next_state': []}
    terminate = False
    truncated = False
    state, info = env.reset()
    done = False
    with tqdm(total=10000, desc="Training Episodes") as pbar:
        for episode in range(10000):
            while not done:
                trajectory['state'].append(state)  # int[1]
                action, log_prob = agent.take_action(state)
                state, reward, terminate, truncated, info = env.step(action)
                done = terminate or truncated
                if reward == 1:
                    # 找到目标
                    reward = 10
                    print("Episode:", episode, "Finished")
                elif terminate == True:
                    # 掉进冰湖
                    reward = -10
                else:
                    reward = -0.5
                trajectory['action'].append(action)
                trajectory['reward'].append(reward)
                trajectory['next_state'].append(state)
                
            losses_data, perf_data = agent.update(trajectory)

            if write_data == True and episode % summary_window == 0:
                write_to_tensorboard(writer, episode, losses_data, perf_data)

            trajectory = {'state': [], 'action': [], 'reward': [], 'next_state': []}
            state, info = env.reset()
            done = False
            pbar.set_postfix({
                'Reward': f"{perf_data['reward']:.2f}",
                'Success Rate': f"{perf_data['success_rate']:.2f}",
                'Steps': perf_data['steps']
            })
            pbar.update(1)

