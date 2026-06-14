import gymnasium as gym
from ale_py import ALEInterface

# 现在可以获取所有注册的Atari环境
all_envs = gym.envs.registry.keys()
atari_envs = [env_id for env_id in all_envs if "ALE/" in env_id]

print("\n已注册的Atari环境:")
for env_id in sorted(atari_envs):
    print(f" - {env_id}")