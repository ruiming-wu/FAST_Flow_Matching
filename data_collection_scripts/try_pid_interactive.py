"""
This script allows interactive testing of PID controller parameters on the MuJoCo InvertedPendulum-v5 environment.
Features:
- Simulates 1-second trajectories using given PID gains (Kp, Ki, Kd)
- Reconstructs MuJoCo's internal termination rule (|angle| > 0.2 rad) to determine "stability"
- Visualizes both pendulum angle and control force
"""
import matplotlib.pyplot as plt
from pid_controller import PIDController
from sample_trajectory import sample_trajectory
import gymnasium as gym
import numpy as np
import os

def plot_trajectory(traj):
    t = np.arange(len(traj["observations"])) * 0.02
    angles = [obs[1] for obs in traj["observations"]]
    
    actions = [a[0] for a in traj["actions"]]

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.axhline(y=0.2, color='r', linestyle='--', linewidth=0.8)
    plt.axhline(y=-0.2, color='r', linestyle='--', linewidth=0.8)
    plt.plot(t, angles)
    plt.title("Pendulum Angle Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (rad)")

    plt.subplot(1, 2, 2)
    plt.plot(t, actions)
    plt.title("Action Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Force")
    plt.tight_layout()
    plt.show()

def check_stability(traj, angle_threshold=0.2, expected_steps=250):
    angles = np.array([obs[1] for obs in traj["observations"]])
    
    if len(angles) < expected_steps:
        print(f"✗ Trajectory ended early: only {len(angles)} steps.")
        return False

    exceed_mask = np.abs(angles) > angle_threshold
    if np.any(exceed_mask):
        first_exceed_step = np.argmax(exceed_mask)
        max_angle = np.max(np.abs(angles))
        print(f"✗ Angle exceeded ±{angle_threshold:.2f} rad at step {first_exceed_step} (max = {max_angle:.4f} rad).")
        return False

    print("This trajectory is STABLE under current criteria.")
    return True


def main():
    Kp = float(input("Enter Kp: "))
    Ki = float(input("Enter Ki: "))
    Kd = float(input("Enter Kd: "))

    pid = PIDController(Kp, Ki, Kd)
    env = gym.make("InvertedPendulum-v5")
    init_state = {
    "qpos": np.array([0.0, 0.00]),
    "qvel": np.array([0.0, 0.00]),
    }
    traj = sample_trajectory(env, pid, init_state=init_state)

    check_stability(traj)
    plot_trajectory(traj)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    traj_dir = os.path.join(script_dir, "data", "trajs")
    os.makedirs(traj_dir, exist_ok=True)
    
    decision = input("Save this trajectory? (y/n): ").lower()
    if decision == 'y':

        filename = f"manual_Kp{Kp}_Ki{Ki}_Kd{Kd}.npz"
        filepath = os.path.join(traj_dir, filename)
        np.savez(filepath, **traj, pid=np.array([Kp, Ki, Kd]))
        #print(f"Trajectory saved to {filepath}")

        param_dir = os.path.join(script_dir, "data", "parameters")
        os.makedirs(param_dir, exist_ok=True)

        param_file = os.path.join(param_dir, f"manual_Kp{Kp:.1f}_Kd{Kd:.1f}.txt")
        with open(param_file, "a") as f:
            f.write(f"Kp={Kp}, Ki={Ki}, Kd={Kd}\n")
if __name__ == "__main__":
    main()
