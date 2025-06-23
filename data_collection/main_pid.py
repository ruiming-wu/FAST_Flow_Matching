import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from data_collection.env_pid import run_pid_in_env

def save_trajectory(traj, act, idx, out_dir):
    # traj: (N, 4), act: (N,)
    os.makedirs(out_dir, exist_ok=True)
    # np.save(os.path.join(out_dir, f"{idx:04d}_{seg+1}.npy"), np.hstack([traj, act.reshape(-1, 1)]))
    np.save(os.path.join(out_dir, f"{idx:04d}.npy"), np.hstack([traj, act.reshape(-1, 1)]))

def save_params(params, init_obs, idx, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{idx:04d}.txt"), "w") as f:
        for k, v in params.items():
            f.write(f"{k}: {v}\n")
        f.write("init_obs: " + np.array2string(init_obs, separator=', ') + "\n")

def save_plot(fig_obj, idx, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fig_obj.savefig(os.path.join(out_dir, f"{idx:04d}.png"))
    plt.close(fig_obj)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, default=1, help="Trajectory index for saving files")
    args = parser.parse_args()
    idx = args.idx

    # 1. 随机扰动PID参数
    base_params = {
        "Kp1": -3.0, "Ki1": -0.01, "Kd1": -0.15,
        "Kp2": 0.08, "Ki2": 0.05, "Kd2": 0.001
    }
    scale1 = np.random.uniform(0.9, 1.1)
    scale2 = np.random.uniform(0.9, 1.1)
    scale3 = np.random.uniform(0.9, 1.1)
    scale4 = np.random.uniform(0.9, 1.1)  
    scale5 = np.random.uniform(0.9, 1.1)
    scale6 = np.random.uniform(0.9, 1.1)
    pid_params = {
        "Kp1": base_params["Kp1"] * scale1,
        "Ki1": base_params["Ki1"] * scale2,
        "Kd1": base_params["Kd1"] * scale3,
        "Kp2": base_params["Kp2"] * scale4,
        "Ki2": base_params["Ki2"] * scale5,
        "Kd2": base_params["Kd2"] * scale6,
    }

    # 2. 采集轨迹
    obs_arr, act_arr, plt_obj = run_pid_in_env(
        Kp1=pid_params["Kp1"], Ki1=pid_params["Ki1"], Kd1=pid_params["Kd1"],
        Kp2=pid_params["Kp2"], Ki2=pid_params["Ki2"], Kd2=pid_params["Kd2"],
        render=False,
        rand_init=True,
        rand_init_scale=0.5,
        max_steps=100,
        time_sleep=0.0
    )

    # 3. 保存轨迹
    save_trajectory(obs_arr, act_arr, idx, "data/trajs")

    # 4. 保存参数和初始状态
    save_params(pid_params, obs_arr[0], idx, "data/params")

    # 5. 保存图片
    save_plot(plt_obj, idx, "data/trajs_pics")