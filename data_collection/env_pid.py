import time
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from data_collection.pid import PIDController

def run_pid_in_env(
    Kp1=1.0, Ki1=0.0, Kd1=0.0,
    Kp2=1.0, Ki2=0.0, Kd2=0.0,
    render=False,
    rand_init=False,
    max_steps=1000,
    time_sleep=0.02
):
    env = gym.make("InvertedPendulum-v5", reset_noise_scale=0.0, render_mode="human" if render else None)
    obs, _ = env.reset()
    if rand_init:
        qpos = np.random.uniform(-0.5, 0.5, size=env.unwrapped.data.qpos.shape)
        qvel = np.random.uniform(-0.5, 0.5, size=env.unwrapped.data.qvel.shape)
        env.unwrapped.set_state(qpos, qvel)
    else:
        env.unwrapped.set_state(np.array([0.0, 0.0]), np.array([0.0, 0.0]))
    obs = np.concatenate([env.unwrapped.data.qpos, env.unwrapped.data.qvel]).ravel()
    print("Initial observation:", obs)

    angel_pid = PIDController(Kp=Kp1, Ki=Ki1, Kd=Kd1)
    vel_pid = PIDController(Kp=Kp2, Ki=Ki2, Kd=Kd2)
    angel_pid.reset()
    vel_pid.reset()

    obs_list, theta_ref_list, act_list, reward_list = [], [], [], []
    t = 0
    done = False

    while not done and t < max_steps:
        pos = obs[0]
        theta = obs[1]
        vel = obs[2]
        theta_dot = obs[3]
        theta_ref = np.clip(vel_pid.get_action(vel, set_value=0.0, dt=0.02), -0.3, 0.3)
        action = np.clip(angel_pid.get_action(theta, set_value=theta_ref, dt=0.02), -3.0, 3.0)
        action = np.array(action).reshape(-1)
        act_list.append(action)

        obs_list.append(obs)        
        if t % 200 == 0 and t != 0:
            action += np.random.uniform(-3.0, 3.0, size=action.shape)
        action += np.random.uniform(-0.02, 0.02, size=action.shape)
        action = np.clip(action, -3.0, 3.0)
        theta_ref_list.append(theta_ref)
        obs, reward, terminated, truncated, _ = env.step(action)
        reward_list.append(reward)
        t += 1
        if render:
            env.render()
        if terminated or truncated:
            print(f"Episode finished at step {t}")
            done = True
        time.sleep(time_sleep)

    env.close()

    obs_arr = np.array(obs_list)
    act_list = np.array(act_list).reshape(-1)
    theta_ref_list = np.array(theta_ref_list).reshape(-1)
    reward_arr = np.array(reward_list)

    # 可视化
    fig = plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(obs_arr[:, 1], label="theta")
    plt.plot(obs_arr[:, 3], label="theta_dot")
    plt.title("Pendulum Angle and Angular Velocity")
    plt.plot(theta_ref_list, label="Theta Reference", linestyle='--')
    plt.xlabel("Step")
    plt.ylabel("Angle / Angular Velocity")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(obs_arr[:, 0], label="pos")
    plt.plot(obs_arr[:, 2], label="vel")
    plt.title("Cart Position and Velocity")
    plt.xlabel("Step")
    plt.ylabel("Position / Velocity")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(act_list, label="PID Action")
    plt.title("PID Action Sequence")
    plt.xlabel("Step")
    plt.ylabel("Action")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    print("Total reward:", reward_arr.sum())
    print("Episode length:", len(obs_arr))

    return obs_arr, act_list, fig

if __name__ == "__main__":
    _, _ = run_pid_in_env(
        Kp1=-3.0, Ki1=-0.01, Kd1=-0.15,
        Kp2=0.08, Ki2=0.05, Kd2=0.0,
        render=True,
        rand_init=True,
        max_steps=1000,
        time_sleep=0.0
    )