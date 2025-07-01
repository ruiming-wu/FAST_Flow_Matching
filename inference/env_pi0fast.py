import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import gymnasium as gym

from inference.inference_pi0fast import infer_pi0fast_token_sequence
from fast.decoder import decoder
from scipy.fftpack import idct
from utils.build_corpus import GAMMA

def run_pi0fast_in_env(
    model_path,
    tokenizer_path,
    chunk_len=50,
    max_seq_len=25,
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
        # 重新推理token序列
        state_vec = obs[:4]
        tokens = infer_pi0fast_token_sequence(
            model_path,
            state_vec,
            max_seq_len=max_seq_len,
            device=device
        )  # 含BOS/EOS

        # 解码为int序列
        token_ints = decoder(tokens, tokenizer_path)
        # 补齐到chunk_len
        if len(token_ints) < chunk_len:
            token_ints = token_ints + [0] * (chunk_len - len(token_ints))
        token_ints = token_ints[:chunk_len]

        # 逆量化与逆DCT
        quantized = np.array(token_ints) / GAMMA
        pred_action = idct(quantized, norm='ortho')[:chunk_len]

        for i in range(replan_interval):
            action = np.array([pred_action[i]])
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
    act_arr = np.array(act_list).squeeze(-1)

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
    model_path = "train/trained_models/tinypi0fast_20250624_0911.pth"
    tokenizer_path = "fast/tokenizer/fast_tokenizer.json"
    run_pi0fast_in_env(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        chunk_len=50,
        max_seq_len=25,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        render=True,
        rand_init=True,
        rand_init_scale=0.5,
        replan_interval=10,
        max_steps=200,
        time_sleep=0.02
    )