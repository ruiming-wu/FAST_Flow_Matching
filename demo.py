import gym

def get_cartpole_tau(env_name="CartPole-v1"):
    env = gym.make(env_name)

    # 尝试直接访问 tau 属性
    if hasattr(env.unwrapped, 'tau'):
        tau = env.unwrapped.tau
        print(f"Detected control interval (tau): {tau} seconds")
        return tau
    else:
        print("This environment does not expose 'tau'.")
        return None

# 用法
get_cartpole_tau()
