import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # 加载轨迹文件
    data = np.load("data/trajs/00001.npy")  # shape: (200, 5)
    obs = data[:, :4]  # 4维状态
    act = data[:, 4]   # 1维动作
    print("Observation shape:", obs.shape)
    print("Action shape:", act.shape)

    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(obs[:, 1], label="theta")
    plt.plot(obs[:, 3], label="theta_dot")
    plt.title("Pendulum Angle and Angular Velocity")
    plt.xlabel("Step")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(obs[:, 0], label="pos")
    plt.plot(obs[:, 2], label="vel")
    plt.title("Cart Position and Velocity")
    plt.xlabel("Step")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(act, label="Action")
    plt.title("Action Sequence")
    plt.xlabel("Step")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

