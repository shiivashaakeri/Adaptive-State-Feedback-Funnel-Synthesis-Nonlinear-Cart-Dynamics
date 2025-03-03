import numpy as np


class PIDController:
    def __init__(self, Kp, Ki, Kd, dt, min_control, max_control):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.min_control = min_control
        self.max_control = max_control
        self.integral = 0.0
        self.prev_error = 0.0

    def compute_control(self, error):
        # Update the integral term
        self.integral += error * self.dt
        # Compute derivative (rate of change of error)
        derivative = (error - self.prev_error) / self.dt
        # PID formula
        u = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        # Saturate the control signal
        u = np.clip(u, self.min_control, self.max_control)
        self.prev_error = error
        return u
