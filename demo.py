import gym
import numpy as np

# 创建环境
env = gym.make('Pendulum-v1')
obs = env.reset()

for _ in range(2000):
    action = env.action_space.sample()

    # 与环境交互
    obs, reward, done, info = env.step(action)
    env.render()

env.close()
