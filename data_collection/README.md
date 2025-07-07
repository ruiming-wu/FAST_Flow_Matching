# 📦 data_collection Module

This directory provides scripts and tools for collecting trajectories in the Inverted Pendulum MuJoCo environment, supporting PID control and batch data generation for downstream model training and evaluation.

---

## Environment Modifications

To prevent early truncation during trajectory generation, following modifications were made:

- **Angle threshold**: increased from ±0.2 rad → **±0.5 rad**
- **Cart position range**: from [-1.0, 1.0] → **[-10.0, 10.0]**

---

## Main Files & Their Functions

- **pid.py**  
  - Purpose: General PID controller class, supports arbitrary setpoints and can be used for nested control.
  - Key Methods:
    - `__init__(Kp, Ki, Kd)`: Initialize PID gains.
    - `reset()`: Reset integral and previous error.
    - `get_action(current_value, set_value, dt)`: Returns the control output as a numpy array.

- **env_pid.py**  
  - Purpose: Interacts with the Gymnasium MuJoCo `InvertedPendulum-v5` environment using (nested) PID controllers, collects trajectories, and provides visualization.
  - Key Parameters:
    - `Kp1, Ki1, Kd1`: PID parameters for the angle (inner loop)
    - `Kp2, Ki2, Kd2`: PID parameters for the velocity (outer loop)
    - `rand_init`: Whether to randomize the initial state (range: ±0.5)
    - `max_steps`: Number of steps per trajectory
    - `time_sleep`: Simulation step interval (can be set to 0 for fast collection)
  - Returns:  
    - `obs_arr`: Collected state trajectory (N, 4)
    - `act_arr`: Collected action sequence (N,)
    - `fig`: Matplotlib figure object for visualization

- **main_pid.py**  
  - Purpose: Main script for batch trajectory collection, supports command-line arguments, and automatically saves trajectories, parameters, and plots.
  - Key Features:
    - Randomly perturbs PID parameters for each trajectory.
    - Saves trajectory data (`.npy`), PID parameters and initial state (`.txt`), and trajectory plots (`.png`) to corresponding folders.
    - Usage example:
      ```bash
      python -m data_collection.main_pid --idx 1
      ```

---

## Data Storage Format

- **Trajectory Data**  
  - Path: `data/trajs/`
  - Format: Each `.npy` file is a shape=(N, 5) array (Normally N=100); first 4 columns are state (position, angle, velocity, angular velocity), last column is action

- **Parameters & Initial State**  
  - Path: `data/params/`
  - Format: Each `.txt` file records the PID parameters and initial observation for the trajectory

- **Visualization Images**  
  - Path: `data/trajs_pics/`
  - Format: Each `.png` file shows the time series of states and actions for the trajectory

---

## Typical Usage

```bash
# Collect a single trajectory
python -m data_collection.main_pid --idx 1

# Batch collection (use script in ./utils/, need to change the range manually in this script)
python -m utils.batch_collection