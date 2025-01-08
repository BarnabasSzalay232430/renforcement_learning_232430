class PIDController:
    def __init__(self, Kp, Ki, Kd, dt, integral_limit=None):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.integral = 0.0
        self.previous_error = 0.0
        self.integral_limit = integral_limit  # Maximum allowable value for the integral term
        self.setpoint = 0.0  # Initialize the setpoint attribute

    def compute(self, current_value):
        error = self.setpoint - current_value
        proportional = self.Kp * error

        # Update integral term with anti-windup
        self.integral += error * self.dt
        if self.integral_limit is not None:
            self.integral = max(min(self.integral, self.integral_limit), -self.integral_limit)

        integral = self.Ki * self.integral
        derivative = self.Kd * (error - self.previous_error) / self.dt
        self.previous_error = error

        return proportional + integral + derivative

    def get_pid_components(self, current_value):
        """
        Returns the individual PID components: P, I, D
        """
        error = self.setpoint - current_value
        proportional = self.Kp * error
        integral = self.Ki * self.integral  # Use the existing integral state
        derivative = self.Kd * (error - self.previous_error) / self.dt
        return proportional, integral, derivative
