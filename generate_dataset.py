import os
import numpy as np
import gymnasium as gym
from pid_controller import PIDController
from sample_trajectory import sample_trajectory

# ========== Configuration Parameters ==========
base_pid = [20,1.1]  # Only Kp and Kd are used (PD controller), we manually set them up 
N_pid_variants = 20     # Number of perturbed PID variants per base PID
N_init_variants = 5     # Number of initial state variants per PID variant
save_dir = "data/trajs"
os.makedirs(save_dir, exist_ok=True)

summary_log = []  # To store summary info for all saved trajectories

def generate_perturbed_pid(Kp, Kd, scale=0.1):
    """Generate perturbed PID parameters (Ki is fixed at 0)"""
    perturbed_Kp = Kp * np.random.uniform(1 - scale, 1 + scale)
    perturbed_Kd = Kd * np.random.uniform(1 - scale, 1 + scale)
    perturbed_Kd = max(perturbed_Kd, 1e-4)  # Prevent negative or zero derivative gain
    return PIDController(Kp=perturbed_Kp, Ki=0.0, Kd=perturbed_Kd), perturbed_Kp, perturbed_Kd

def generate_init_state():
    """Generate a random initial state with small perturbations"""
    theta = np.random.uniform(-0.01, 0.01)
    theta_dot = np.random.uniform(-0.01, 0.01)
    return {
        "qpos": np.array([0.0, theta]),
        "qvel": np.array([0.0, theta_dot])
    }, theta, theta_dot

def check_stability(traj, angle_threshold=0.2, expected_steps=250):
    """Determine if the trajectory is stable under given criteria"""
    angles = np.array([obs[1] for obs in traj["observations"]])

    if len(angles) < expected_steps:
        return False

    exceed_mask = np.abs(angles) > angle_threshold
    if np.any(exceed_mask):
        return False

    return True

# ========== Main Execution ==========

env = gym.make("InvertedPendulum-v5", render_mode=None)
env.reset()

counter = 0
for i in range(N_pid_variants):
    pid_controller, kp_val, kd_val = generate_perturbed_pid(*base_pid)

    for j in range(N_init_variants):
        init_state, theta, theta_dot = generate_init_state()
        traj = sample_trajectory(env, pid_controller, init_state=init_state)

        if check_stability(traj):
            filename = f"pid_Kp{kp_val:.2f}_Kd{kd_val:.2f}_th{theta:.3f}_dth{theta_dot:.3f}_run{j+1:02d}.npz"
            filepath = os.path.join(save_dir, filename)
            np.savez(filepath,
                     observations=traj["observations"],
                     actions=traj["actions"],
                     rewards=traj["rewards"],
                     pid=np.array([kp_val, 0.0, kd_val]))
            summary_log.append((kp_val, kd_val, theta, theta_dot))
            counter += 1

print("Total stable trajectories saved:", counter)
for idx, (kp, kd, theta, theta_dot) in enumerate(summary_log):
    print(f"  [{idx+1:03}] Kp={kp:.4f}, Kd={kd:.4f}, init_theta={theta:.4f}, init_theta_dot={theta_dot:.4f}")
