import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

if __name__ == "__main__":

    env = gym.make("InvertedPendulum-v5", reset_noise_scale=0.0, render_mode=None)
    obs, _ = env.reset(options={"qpos": np.array([0, -1e-6]), "qvel": np.array([1e-6, -1e-6])})
    print("Initial observation:", obs)

    obs_list, reward_list = [], []
    t = 0
    done = False

    while not done and t < 200:
        action = np.zeros(1)
        obs_list.append(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        reward_list.append(reward)
        t += 1
        if terminated or truncated:
            print(f"Episode finished at step {t}")
            done = True

    env.close()

    obs_arr = np.array(obs_list)
    reward_arr = np.array(reward_list)
    print("Final state:", obs_arr[-1])
    print("Total reward:", reward_arr.sum())
    print("Episode length:", len(obs_arr))

    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.plot(obs_arr[:, 1], label="theta")
    plt.plot(obs_arr[:, 3], label="theta_dot")
    plt.title("Pendulum Angle and Angular Velocity")
    plt.xlabel("Step")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(reward_arr, label="reward")
    plt.title("Reward per Step")
    plt.xlabel("Step")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

