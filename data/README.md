# data Folder Description

This folder contains 4000 collected trajectories for the Inverted Pendulum environment. Each trajectory consists of 100 time steps and has been verified to be valid and stable at the end (the mean angle value of the last 25 steps is within ±0.01 radians).

## Data Generation Details

- **Parameter Randomization:**  
  PID parameters are randomized by scaling the following base values by a factor uniformly sampled from 0.9 to 1.1:
  - `Kp1`: -3.0, `Ki1`: -0.01, `Kd1`: -0.15
  - `Kp2`: 0.08, `Ki2`: 0.05, `Kd2`: 0.0

- **Initial State:**  
  The 4-dimensional initial state is uniformly sampled within ±0.5 for each dimension.

- **PID Output Noise:**  
  Before being applied to the simulation environment, each PID output is perturbed with uniform noise in the range of ±0.02 N.  
  *Note: This noise is not recorded in the actions of `trajs` files, but is intended to encourage the model to learn noise robustness.*

## Data Quality

- All 4000 trajectories are valid and meet the stability criterion (last 25 steps' mean angle within ±0.01 radians).
- During collection, there were 6 failed control cases (early termination by MuJoCo) and 17 cases where the end was not stable (exceeded the ±0.01 radian limit).  
  These trajectories were replaced to ensure all data meets the requirements.

## File Structure

- `trajs/`: Contains the trajectory `.npy` files (shape: 100×5, 4D state + 1D action per step).
- `params/`: Contains the PID parameters and initial state for each trajectory.
- `trajs_pics/`: Contains visualization images for each trajectory.