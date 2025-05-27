import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from pid_controller import PIDController

# Pid parameters
Kp, Ki, Kd = 4.0, 0.0, 0.0
dt = 0.02
theta_limit = 0.2
x_limit = 2.4

#initialiazition with MuJoCo environment
env = gym.make("InvertedPendulum-v5", render_mode=None)
#print(env.spec.max_episode_steps)
#env._max_episode_steps = 1000  
pid = PIDController(Kp, Ki, Kd)
obs, info = env.reset(seed=42)

# data initialization
history = {
    'step': [0],
    'theta': [obs[2]],
    'x': [obs[0]],
    'action': [0.0]
}

# Processing
for step in range(1, 1001):
    # control implementation
    error = 0.0 - history['theta'][-1]
    action = np.clip(pid.compute(error, dt), -3.0, 3.0)
    
    # act, and update
    obs, _, terminated, truncated, _ = env.step([action])
    
    # data
    history['step'].append(step)
    history['theta'].append(obs[2])
    history['x'].append(obs[0])
    history['action'].append(action)
    
    # Check termination
    if terminated or truncated:
        print(f"terminate at step {step}: theta={obs[2]:.4f}, x={obs[0]:.4f}")
        break

env.close()


theta_arr = np.array(history['theta'])
x_arr = np.array(history['x'])
steps = np.array(history['step']) * dt
print(info)

first_theta_over = np.where(np.abs(theta_arr) > theta_limit)[0]
first_x_over = np.where(np.abs(x_arr) > x_limit)[0]

print(f"step={step}, theta={obs[2]:.4f}")
plt.figure(figsize=(12, 8))


plt.subplot(3, 1, 1)
plt.plot(steps, theta_arr, label='θ (rad)')
plt.axhline(theta_limit, c='r', ls='--', label='Boundary')
plt.axhline(-theta_limit, c='r', ls='--')
if len(first_theta_over) > 0:
    plt.axvline(steps[first_theta_over[0]], c='purple', ls=':', label='First time')
plt.ylabel("Angle")
plt.legend()


plt.subplot(3, 1, 2)
plt.plot(steps, history['action'])
plt.ylabel("Force")


plt.subplot(3, 1, 3)
plt.plot(steps, x_arr, label='x (m)')
plt.axhline(x_limit, c='r', ls='--')
plt.axhline(-x_limit, c='r', ls='--')
if len(first_x_over) > 0:
    plt.axvline(steps[first_x_over[0]], c='purple', ls=':')
plt.xlabel("time (s)")
plt.ylabel("x")
plt.tight_layout()
plt.show()