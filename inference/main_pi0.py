import os
import numpy as np
import torch
import gymnasium as gym
from tqdm import tqdm

from inference.inference_pi0 import infer_pi0_action_sequence

def evaluate_models_on_env(
    model_dir="train/trained_models",
    model_prefix="tinypi0_",
    total_runs=1000,
    steps_per_run=100,
    chunk_len=50,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    rand_init=True,
    rand_init_scale=0.5,
    rand_noise_scale=0.0,
    replan_interval=10,
    verbose=True
):
    # 1. 收集所有模型
    model_files = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.startswith(model_prefix) and f.endswith(".pth")]
    model_files.sort()
    n_models = len(model_files)
    if n_models == 0:
        raise RuntimeError("No model found!")
    runs_per_model = total_runs // n_models
    if verbose:
        print(f"Found {n_models} models, each will run {runs_per_model} times.")

    # 2. 记录结果
    all_theta = []  # 每次模拟的theta轨迹 (run_idx, steps)
    all_theta_dot = []
    all_cart_vel = []  # 新增：每轮cart速度序列
    all_success = []  # 1=成功, 0=失败
    all_converge_step = []
    all_model_idx = []

    for model_idx, model_path in enumerate(model_files):
        if verbose:
            print(f"Evaluating model {model_path} ...")
        for run in tqdm(range(runs_per_model), disable=not verbose):
            # 环境初始化
            env = gym.make("InvertedPendulum-v5", reset_noise_scale=rand_init_scale if rand_init else 0.0)
            obs, _ = env.reset()
            theta_traj = []
            theta_dot_traj = []
            cart_vel_traj = []
            done = False
            t = 0
            while not done and t < steps_per_run:
                # 推理动作序列
                state_vec = obs[:4]
                actions = infer_pi0_action_sequence(
                    model_path,
                    state_vec,
                    chunk_len=chunk_len,
                    device=device
                )  # (chunk_len, 1)
                for i in range(replan_interval):
                    action = actions[i]
                    action += np.random.uniform(-rand_noise_scale, rand_noise_scale, size=action.shape)
                    action = np.clip(action, -3.0, 3.0)
                    obs, _, terminated, truncated, _ = env.step(action)
                    theta_traj.append(obs[1])
                    theta_dot_traj.append(obs[3])
                    cart_vel_traj.append(obs[2])
                    t += 1
                    if terminated or truncated or t >= steps_per_run:
                        done = True
                        break
            env.close()
            # 补齐长度
            if len(theta_traj) < steps_per_run:
                theta_traj += [np.nan] * (steps_per_run - len(theta_traj))
                theta_dot_traj += [np.nan] * (steps_per_run - len(theta_dot_traj))
            if len(cart_vel_traj) < steps_per_run:
                cart_vel_traj += [np.nan] * (steps_per_run - len(cart_vel_traj))
            all_theta.append(theta_traj)
            all_theta_dot.append(theta_dot_traj)
            all_cart_vel.append(cart_vel_traj)
            all_success.append(int(t == steps_per_run))
            all_model_idx.append(model_idx)
            # 收敛步数
            converge_step = get_converge_step_from_theta(np.array(theta_traj), max_len=steps_per_run)
            all_converge_step.append(converge_step)

    # 3. 汇总为array
    all_theta = np.array(all_theta)  # (total_runs, steps_per_run)
    all_theta_dot = np.array(all_theta_dot)
    all_cart_vel = np.array(all_cart_vel)
    all_success = np.array(all_success)
    all_converge_step = np.array(all_converge_step)
    print('Converge steps:', all_converge_step)
    all_model_idx = np.array(all_model_idx)

    # 4. 统计指标
    success_rate = np.mean(all_success)
    avg_converge_step = np.nanmean(all_converge_step)
    converge_rate = np.mean(all_converge_step < steps_per_run)
    mean_cart_vel = np.nanmean(all_cart_vel)
    std_cart_vel = np.nanstd(all_cart_vel)

    print(f"Total runs: {total_runs}")
    print(f"Success rate (full {steps_per_run} steps): {success_rate*100:.2f}%")
    print(f"Converge rate (theta): {converge_rate*100:.2f}%")
    print(f"Average converge step: {avg_converge_step:.2f}")
    print(f"cart velocity over all runs: {mean_cart_vel:.4f} ± {std_cart_vel:.4f}")

    # 每个模型的表现
    for i, model_path in enumerate(model_files):
        model_mask = (all_model_idx == i)
        print(f"Model {os.path.basename(model_path)}: success {all_success[model_mask].mean()*100:.1f}%, "
              f"converge {np.mean(all_converge_step[model_mask] < steps_per_run)*100:.1f}%, "
              f"avg converge step {np.nanmean(all_converge_step[model_mask]):.1f}, "
              f"cart vel {np.nanmean(all_cart_vel[model_mask]):.4f} ± {np.nanstd(all_cart_vel[model_mask]):.4f}")

    # 统计平均收敛步数（只统计真正收敛的）
    valid_converge = all_converge_step[all_converge_step < steps_per_run]
    if len(valid_converge) > 0:
        avg_converge_step = np.nanmean(valid_converge)
        print(f"Average converge step (only converged): {avg_converge_step:.2f}")
    else:
        print("No successful convergence in any run.")

    # 返回所有数据，便于后续分析
    return {
        "theta": all_theta,
        "theta_dot": all_theta_dot,
        "cart_vel": all_cart_vel,
        "success": all_success,
        "converge_step": all_converge_step,
        "model_idx": all_model_idx,
        "model_files": model_files
    }

def get_converge_step_from_theta(theta_arr, max_len, threshold=0.02, min_len=10):
    # theta_arr: (steps,)
    if len(theta_arr) < min_len:
        return np.nan  # 不足最小长度，无法判断收敛
    for i in range(len(theta_arr) - min_len + 1):
        window = theta_arr[i:i+min_len]
        if np.all(np.abs(window) < threshold):
            return i
    return np.nan

if __name__ == "__main__":
    results = evaluate_models_on_env(
        model_dir="train/trained_models",
        model_prefix="tinypi0_",
        total_runs=1000,
        steps_per_run=100,
        chunk_len=50,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        rand_init=True,
        rand_init_scale=0.5,
        rand_noise_scale=0.05,
        replan_interval=13,
        verbose=True
    )