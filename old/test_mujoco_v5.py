import gymnasium as gym
import numpy as np

# 创建 v5 环境
env = gym.make("InvertedPendulum-v5", render_mode="human")  # 你也可以用 render_mode=None 加快运行

# 重置环境，并设定随机种子
obs, info = env.reset(seed=42)

# 模拟 1000 个时间步
for _ in range(1000):
    action = env.action_space.sample()  # 随机动作（可替换为 PID 控制）
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()

env.close()
