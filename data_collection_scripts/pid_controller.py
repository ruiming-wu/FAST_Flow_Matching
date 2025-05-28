class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0.0
        self.prev_error = 0.0
        
    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.first_call = True

    def compute(self, error, dt):
        if self.first_call:
            self.prev_error = error
            self.first_call = False
            derivative = 0.0  
        else:
            derivative = (error - self.prev_error) / dt
        self.integral += error * dt

        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative 
        self.prev_error = error
        return output
