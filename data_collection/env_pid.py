import time
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from data_collection.pid import PIDController

def run_pid_in_env(
    Kp1=0.0, Ki1=0.0, Kd1=0.0,
    Kp2=0.0, Ki2=0.0, Kd2=0.0,
    render=False,
    rand_init=False,
    rand_init_scale=0.5,
    max_steps=200,
    time_sleep=0.02
):
    if rand_init:
        env = gym.make("InvertedPendulum-v5", reset_noise_scale=rand_init_scale, render_mode="human" if render else None)
    else:
        env = gym.make("InvertedPendulum-v5", reset_noise_scale=0.0, render_mode="human" if render else None)
    obs, _ = env.reset()

    print("Initial observation:", obs)

    vel_pid = PIDController(Kp=Kp2, Ki=Ki2, Kd=Kd2)
    angel_pid = PIDController(Kp=Kp1, Ki=Ki1, Kd=Kd1)
    vel_pid.reset()
    angel_pid.reset()

    obs_list, act_list, theta_ref_list = [], [], []
    t = 0
    done = False

    while not done and t < max_steps:
        theta = obs[1]
        vel = obs[2]

        theta_ref = np.clip(vel_pid.get_action(vel, set_value=0.0, dt=0.02), -np.pi / 12.0, np.pi / 12.0)
        action = np.clip(angel_pid.get_action(theta, set_value=theta_ref-0.00166666, dt=0.02), -3.0, 3.0)

        obs_list.append(obs)
        act_list.append(action.item())
        theta_ref_list.append(theta_ref.item())
      
        if t % 200 == 0 and t != 0:
            action += np.random.uniform(-3.0, 3.0, size=action.shape)
        action += np.random.uniform(-0.05, 0.05, size=action.shape)
        action = np.clip(action, -3.0, 3.0)

        obs, _ , terminated, truncated, _ = env.step(action)
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

    fig = plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(obs_arr[:, 1], label="theta")
    plt.plot(obs_arr[:, 3], label="theta_dot")
    plt.plot(theta_ref_list, label="Theta Reference", linestyle='--')
    plt.title("Pendulum Angle and Angular Velocity")
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
    plt.plot(act_list, label="PID Action (without noise)")
    plt.title("PID Action Sequence")
    plt.xlabel("Step")
    plt.ylabel("Action")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    return obs_arr, act_list, fig

if __name__ == "__main__":
    run_pid_in_env(
        Kp1=-3.00, Ki1=-0.01, Kd1=-0.15,
        Kp2=0.08, Ki2=0.05, Kd2=0.001,
        render=True,
        rand_init=True,
        rand_init_scale=0.5,
        max_steps=100,
        time_sleep=0.0
    )