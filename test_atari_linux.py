import os
import gymnasium as gym
from ale_py import ALEInterface

def verify_atari():
    try:
        # 1. 检查 ROM 文件是否存在
        # rom_path = os.environ["ALE_PY_ROM_DIR"]
        # breakout_path = os.path.join(rom_path, "breakout.bin")
        # if not os.path.exists(breakout_path):
        #     print(f"❌ Breakout ROM 不存在于: {breakout_path}")
        #     return False
        # print(f"✅ Breakout ROM 存在: {breakout_path}")
        
        # 2. 使用 ALEInterface 加载 ROM
        # ale = ALEInterface()
        # ale.loadROM(breakout_path)
        # print("✅ Breakout ROM 成功加载")
        
        # 3. 创建 Gymnasium 环境
        env = gym.make("ALE/Breakout-v5", render_mode="human")
        print("✅ Gymnasium 环境创建成功")
        
        # 4. 测试环境运行
        observation, info = env.reset()
        print(f"✅ 环境重置成功，观察形状: {observation.shape}")
        
        # 运行几步测试
        for _ in range(1000):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                observation, info = env.reset()
        
        env.close()
        print("✅ 环境运行测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        return False

if __name__ == "__main__":
    if verify_atari():
        print("\n  Atari 环境验证成功！可以开始训练了！")
    else:
        print("\n⚠️ 验证失败，请检查安装")