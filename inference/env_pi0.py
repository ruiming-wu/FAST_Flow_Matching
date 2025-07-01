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
    rand_init=False,
    rand_init_scale=0.5,
    replan_interval=10,
    max_steps=200,
    time_sleep=0.02
):
    if rand_init:
        env = gym.make("InvertedPendulum-v5", reset_noise_scale=rand_init_scale, render_mode="human" if render else None)
    else:
        env = gym.make("InvertedPendulum-v5", reset_noise_scale=0.0, render_mode="human" if render else None)
    # # Wrap the environment to record video every 10 episodes
    # env = gym.wrappers.RecordVideo(env, "./vid", episode_trigger=lambda episode_id: episode_id % max_steps ==  0)
    obs, _ = env.reset()

    print("Initial observation:", obs)

    obs_list, act_list = [], []
    t = 0
    done = False

    while not done and t < max_steps:
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
            obs, _, terminated, truncated, _ = env.step(action)
            t += 1
            if render:
                env.render()
            if terminated or truncated:
                print(f"Episode finished at step {t}")
                done = True
                break
            time.sleep(time_sleep)

    env.close()

    obs_arr = np.array(obs_list)
    act_arr = np.array(act_list)

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

if __name__ == "__main__":
    model_path = "train/trained_models/tinypi0_20250620_0133.pth"
    run_pi0_in_env(
        model_path=model_path,
        chunk_len=50,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        render=True,
        rand_init=True,
        rand_init_scale=0.5,  
        replan_interval=10,
        max_steps=200,
        time_sleep=0.02
    )