import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import gymnasium as gym

from inference.inference_pi0 import infer_pi0_action_sequence

def run_pi0_in_env(
    model_path,
    chunk_len=50,
    device='cpu',
    render=False,
    random_init=True,
    replan_interval=16
):
    # 环境初始化
    env = gym.make("InvertedPendulum-v5", reset_noise_scale=0.1, render_mode="human" if render else None)
    obs, _ = env.reset()
    if random_init:
        qpos = np.random.uniform(-0.5, 0.5, size=env.unwrapped.data.qpos.shape)
        qvel = np.random.uniform(-0.5, 0.5, size=env.unwrapped.data.qvel.shape)
        env.unwrapped.set_state(qpos, qvel)
        obs = np.concatenate([env.unwrapped.data.qpos, env.unwrapped.data.qvel]).ravel()
    else:
        env.unwrapped.set_state(np.array([0.0, 0.0]), np.array([0.0, 0.0]))
        obs = np.concatenate([env.unwrapped.data.qpos, env.unwrapped.data.qvel]).ravel()
    print("Initial observation:", obs)

    obs_list, act_list, reward_list = [], [], []
    t = 0
    done = False

    while not done and t < 1000:  # 最多执行1000步，防止死循环
        # 重新推理动作序列
        state_vec = obs[:4]
        actions = infer_pi0_action_sequence(
            model_path,
            state_vec,
            chunk_len=chunk_len,
            device=device
        )  # (chunk_len, 1)

        for i in range(replan_interval):
            action = actions[i]
            obs_list.append(obs)
            act_list.append(action)
            obs, reward, terminated, truncated, info = env.step(action)
            reward_list.append(reward)
            t += 1
            if render:
                env.render()
            if terminated or truncated:
                print(f"Episode finished at step {t}")
                done = True
                break
            time.sleep(0.05)

    env.close()

    obs_arr = np.array(obs_list)
    act_arr = np.array(act_list)
    reward_arr = np.array(reward_list)

    # 可视化
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(act_arr, marker='o')
    plt.title("Action Sequence")
    plt.xlabel("Step")
    plt.ylabel("Action Value")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(obs_arr[:, 1], label="theta")
    plt.plot(obs_arr[:, 3], label="theta_dot")
    plt.title("Pendulum Angle and Angular Velocity")
    plt.xlabel("Step")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f"Total reward: {reward_arr.sum():.2f}")

if __name__ == "__main__":
    model_path = "train/trained_models/tinypi0_194916062025.pth"
    run_pi0_in_env(
        model_path=model_path,
        chunk_len=50,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        render=True,         # True可视化环境
        random_init=True,    # True为随机初始状态
        replan_interval=10   # 每25步重新推理一次
    )