import numpy as np
import os
import gymnasium as gym

from pid_controller import PIDController
from sample_trajectory import sample_trajectory

def is_stable(traj, angle_threshold=0.2, expected_steps=50):
    """
    Determine if a trajectory is considered stable.
    Criteria:
    - All angles must remain within ±angle_threshold radians
    - Trajectory must reach the full expected length
    """
    angles = traj["observations"][:, 1]
    return len(angles) == expected_steps and np.all(np.abs(angles) < angle_threshold)

def generate_initial_states(env, n=5):
    """
    Generate a list of random initial states with small perturbations.
    Each state includes randomized qpos and qvel within small ranges.
    """
    qpos_range = 0.05
    qvel_range = 0.1
    init_states = []
    for _ in range(n):
        qpos = np.random.uniform(-qpos_range, qpos_range, size=env.unwrapped.model.nq)
        qvel = np.random.uniform(-qvel_range, qvel_range, size=env.unwrapped.model.nv)

        init_states.append({'qpos': qpos, 'qvel': qvel})
    return init_states

def main():
    # === Configuration ===
    use_random_init = False         # Set to True to evaluate each PID under multiple randomized initial states
    num_random_inits = 5            # Number of random initial states per PID (only used if use_random_init is True)

    # === File paths ===
    pid_file = "pid_variants.npy"                 # Input file with all PID candidates
    output_pid_file = "stable_pid_variants.npy"   # Output file for successful PID values
    output_traj_dir = "filtered_trajs"            # Directory to store trajectory .npz files
    os.makedirs(output_traj_dir, exist_ok=True)

    # === Environment setup ===
    env = gym.make("InvertedPendulum-v5")

    # === Load all PID variants to be evaluated ===
    all_pids = np.load(pid_file)
    stable_pids = []
    traj_count = 0

    for i, pid_vals in enumerate(all_pids):
        pid = PIDController(*pid_vals)

        # === Choose initial states ===
        if use_random_init:
            init_states = generate_initial_states(env, n=num_random_inits)
        else:
            init_states = [{"qpos": np.zeros(env.unwrapped.model.nq), "qvel": np.zeros(env.unwrapped.model.nv)}]


        all_success = True
        for init_state in init_states:
            traj = sample_trajectory(env, pid, init_state=init_state)
            if not is_stable(traj):
                all_success = False
                break  # One failure is enough to discard this PID

        # === Save successful trajectory and record the PID ===
        if all_success:
            stable_pids.append(pid_vals)
            save_path = os.path.join(output_traj_dir, f"traj_{traj_count:05d}.npz")
            np.savez(save_path, **traj, pid=np.array(pid_vals))
            traj_count += 1

    # === Save the final list of stable PIDs ===
    if stable_pids:
        np.save(output_pid_file, np.array(stable_pids))
        print(f"\n {len(stable_pids)} stable PID variants saved to: {output_pid_file}")
    else:
        print("\n No stable PID variants found. You may consider adjusting your base PID or perturbation settings.")

    print(f" {traj_count} stable trajectories saved in directory: {output_traj_dir}")

if __name__ == "__main__":
    main()
