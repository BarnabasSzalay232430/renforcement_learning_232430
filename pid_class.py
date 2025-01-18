import numpy as np

class PIDController:
    def __init__(self, Kp, Ki, Kd, dt):
        self.Kp = np.array(Kp)
        self.Ki = np.array(Ki)
        self.Kd = np.array(Kd)
        self.dt = dt
        self.integral = np.zeros(3)
        self.previous_error = np.zeros(3)

    def compute(self, current_value, target_value):
        error = np.array(target_value) - np.array(current_value)
        self.integral += error * self.dt  # Update integral term
        derivative = (error - self.previous_error) / self.dt
        self.previous_error = error
        # PID output
        return self.Kp * error + self.Ki * self.integral + self.Kd * derivative

    def reset(self):
        """Resets the integral and error state"""
        self.integral = np.zeros(3)
        self.previous_error = np.zeros(3)
