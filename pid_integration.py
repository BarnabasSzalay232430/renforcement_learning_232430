import numpy as np
import matplotlib.pyplot as plt
import logging
from pid_class import PIDController

def simulate_pid_response(pid_controller, setpoint, timesteps, accuracy_threshold=0.01):
    dt = pid_controller.dt
    state = np.zeros(3)
    responses = []
    errors = []
    goal_reached = False
    steps_to_goal = None

    for t in range(timesteps):
        error = np.array(setpoint) - state
        control_signal = pid_controller.compute(state, setpoint)
        state += control_signal * dt  # Simplified system dynamics

        responses.append(state.copy())
        errors.append(np.linalg.norm(error))

        if not goal_reached and np.linalg.norm(error) <= accuracy_threshold:
            goal_reached = True
            steps_to_goal = t

    return np.array(responses), np.array(errors), goal_reached, steps_to_goal


def analyze_and_plot_response(responses, setpoint, title):
    timesteps = len(responses)
    time = np.linspace(0, timesteps * 0.1, timesteps)  # Adjust time scaling if needed
    steady_state = setpoint

    metrics = []
    for i in range(3):  # For x, y, z axes
        axis_response = responses[:, i]

        # Calculate rise time
        rise_time = next((t for t, y in enumerate(axis_response) if y >= 0.9 * steady_state[i]), None)

        # Calculate settling time
        settling_time = next(
            (t for t, y in enumerate(axis_response) if np.all(np.abs(axis_response[t:] - steady_state[i]) <= 0.02 * steady_state[i])),
            None
        )

        # Calculate overshoot
        overshoot = np.max(axis_response) - steady_state[i]

        metrics.append((rise_time, settling_time, overshoot))

        # Plot each axis response
        plt.figure(figsize=(6, 4))
        plt.plot(time, axis_response, label=f"Axis {i+1} Response")
        plt.axhline(setpoint[i], color="red", linestyle="--", label="Setpoint")
        plt.axhline(0.9 * steady_state[i], color="green", linestyle=":", label="90% of Setpoint")
        plt.axhline(1.02 * steady_state[i], color="blue", linestyle=":", label="2% Band")
        plt.axhline(0.98 * steady_state[i], color="blue", linestyle=":")
        plt.title(f"{title} - Axis {i+1}")
        plt.xlabel("Time (s)")
        plt.ylabel("Response")
        plt.legend()
        plt.grid()
        plt.show()

    return metrics


def main():
    logging.basicConfig(level=logging.INFO)

    # PID configurations
    timesteps = 200
    setpoint = np.array([1.0, 1.0, 1.0])

    # Slow response (low Kp, no Ki, no Kd)
    pid_slow = PIDController(Kp=[0.5, 0.5, 0.5], Ki=[0.0, 0.0, 0.0], Kd=[0.0, 0.0, 0.0], dt=0.1)
    response_slow, errors_slow, goal_reached_slow, steps_to_goal_slow = simulate_pid_response(pid_slow, setpoint, timesteps)
    metrics_slow = analyze_and_plot_response(response_slow, setpoint, "Slow Response (Low Kp)")

    # Oscillatory response (high Kp, low Kd)
    pid_oscillatory = PIDController(Kp=[2.0, 2.0, 2.0], Ki=[0.0, 0.0, 0.0], Kd=[0.2, 0.2, 0.2], dt=0.1)
    response_oscillatory, errors_oscillatory, goal_reached_oscillatory, steps_to_goal_oscillatory = simulate_pid_response(pid_oscillatory, setpoint, timesteps)
    metrics_oscillatory = analyze_and_plot_response(response_oscillatory, setpoint, "Oscillatory Response (High Kp, Low Kd)")

    # Tuned response (balanced Kp, Ki, Kd)
    pid_tuned = PIDController(Kp=[1.0, 1.0, 1.0], Ki=[0.1, 0.1, 0.1], Kd=[0.5, 0.5, 0.5], dt=0.1)
    response_tuned, errors_tuned, goal_reached_tuned, steps_to_goal_tuned = simulate_pid_response(pid_tuned, setpoint, timesteps)
    metrics_tuned = analyze_and_plot_response(response_tuned, setpoint, "Tuned Response (Balanced Kp, Ki, Kd)")

    # Log results
    logging.info(f"Slow Response Metrics: {metrics_slow}")
    logging.info(f"Slow Response Goal Reached: {goal_reached_slow}, Steps to Goal: {steps_to_goal_slow}")

    logging.info(f"Oscillatory Response Metrics: {metrics_oscillatory}")
    logging.info(f"Oscillatory Response Goal Reached: {goal_reached_oscillatory}, Steps to Goal: {steps_to_goal_oscillatory}")

    logging.info(f"Tuned Response Metrics: {metrics_tuned}")
    logging.info(f"Tuned Response Goal Reached: {goal_reached_tuned}, Steps to Goal: {steps_to_goal_tuned}")


if __name__ == "__main__":
    main()
