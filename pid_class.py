import numpy as np
class PIDController:
    """
    A class for implementing a PID controller for three independent axes (X, Y, Z).

    Attributes:
        Kp (array): Proportional gain for each axis.
        Ki (array): Integral gain for each axis.
        Kd (array): Derivative gain for each axis.
        dt (float): Time step for updates.
        integral (array): Accumulated error for each axis (integral term).
        previous_error (array): Previous error for each axis (used for derivative term).
    """

    def __init__(self, Kp, Ki, Kd, dt):
        """
        Initializes the PID controller with the given gains and time step.

        Args:
            Kp (list or array): Proportional gains for each axis.
            Ki (list or array): Integral gains for each axis.
            Kd (list or array): Derivative gains for each axis.
            dt (float): Time step for the PID updates.
        """
        self.Kp = np.array(Kp)
        self.Ki = np.array(Ki)
        self.Kd = np.array(Kd)
        self.dt = dt
        self.integral = np.zeros(3)
        self.previous_error = np.zeros(3)

    def compute(self, current_value, target_value):
        """
        Computes the control signal based on the PID formula.

        Args:
            current_value (list or array): Current position of the system (X, Y, Z).
            target_value (list or array): Desired position of the system (X, Y, Z).

        Returns:
            array: Control signal to move the system closer to the target position.
        """
        error = np.array(target_value) - np.array(current_value)
        self.integral += error * self.dt  # Update integral term
        derivative = (error - self.previous_error) / self.dt
        self.previous_error = error
        return self.Kp * error + self.Ki * self.integral + self.Kd * derivative

    def reset(self):
        """
        Resets the integral and error states to zero.
        """
        self.integral = np.zeros(3)
        self.previous_error = np.zeros(3)
