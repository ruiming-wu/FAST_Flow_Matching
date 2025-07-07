import numpy as np

class PIDController:
    def __init__(self, Kp=1.0, Ki=0.0, Kd=0.0):
        """
        Kp, Ki, Kd: params for PID controller
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0.0
        self.prev_error = 0.0

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def get_action(self, current_value, set_value, dt=0.02):
        """
        current_value: current measurement (e.g., angle, velocity or position)
        set_value: desired value to reach (e.g., target angle, velocity or position)
        dt: time step for derivative calculation
        return: action to apply
        """
        error = set_value - current_value
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error

        action = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        return np.array(action, dtype=np.float32).reshape(-1)