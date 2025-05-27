import numpy as np
import gymnasium as gym
from pid_controller import PIDController


def sample_trajectory(env, pid: PIDController, init_state=None, max_steps=250, dt=0.02):
    # ==== Initialize states ====
    if init_state is not None:
        env.reset()
        env.unwrapped.set_state(init_state["qpos"], init_state["qvel"])
        obs = np.concatenate([init_state["qpos"], init_state["qvel"]])
        obs, _, _, _, _ = env.step([0.0])
        print("Set initial state manually:")
        print(f"  qpos: {init_state['qpos']}, qvel: {init_state['qvel']}")
    else:
        obs, _ = env.reset()
        print("No init_state provided, using random default state.")
    
    print(f"Initial observation: {obs}")
    print(f"Initial angle (obs[1]): {obs[1]:.4f}\n")

    # ==== PID ====
    pid.reset()
    obs_list, act_list, reward_list = [], [], []

    # ==== Control Loop ====
    for t in range(max_steps):
        angle = obs[1]  # pendulum angle
        error =  angle - 0.0  # target is upright (angle = 0),Note: angle > 0 means the pole leans right, so action should push left (i.e., action < 0) Therefore, we define error = +angle to get correct PID polarity
        action = pid.compute(error, dt) 
        action = np.clip(action, -3.0, 3.0)

        #print(f"[{t:02d}] angle={angle:.4f}, error={error:.4f}, action={action:.4f}")

        obs, reward, terminated, truncated, _ = env.step([action])

        obs_list.append(obs.copy())
        act_list.append([action])
        reward_list.append(reward)

        if terminated:
            print(f"Terminated at step {t+1} due to environment condition (angle too large or invalid state).")
            break
        if truncated:
            print(f"Truncated at step {t+1} (reached max_episode_steps).")
            break

    print(f"\nFinal angle = {obs[1]:.4f} | Trajectory length = {len(obs_list)} steps\n")

    return {
        "observations": np.array(obs_list),
        "actions": np.array(act_list),
        "rewards": np.array(reward_list),
    }
