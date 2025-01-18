import numpy as np
import matplotlib.pyplot as plt
import logging
from pid_class import PIDController
from sim_class import Simulation
import os


# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pid_test2_log.log")
    ]
)

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

print("Current working directory:", os.getcwd())

def plot_response(time_steps, responses, goal_position, title):
    """
    Plots the PID response for visualization.

    Parameters:
        time_steps (list): Time steps of the simulation.
        responses (list): Recorded responses during the simulation.
        goal_position (list): Desired target position [x, y, z].
        title (str): Title of the graph.
    """
    responses = np.array(responses)
    plt.figure(figsize=(10, 6))
    for i, axis in enumerate(['X', 'Y', 'Z']):
        plt.plot(time_steps, responses[:, i], label=f"{axis} Position")
        plt.axhline(goal_position[i], color="red", linestyle="--", label=f"{axis} Setpoint" if i == 0 else None)

    plt.title(title)
    plt.xlabel("Time Steps")
    plt.ylabel("Position")
    plt.legend()
    plt.grid()
    plt.show()

def run_pid_simulation(pid_gains, time_step, goal_position, max_iterations=1000, accuracy_threshold=0.001, hold_duration=50, enable_render=True):
    """
    Simulates the PID controller for a pipette to move to a target position.

    Args:
        pid_gains (dict): PID gains for X, Y, Z axes in the format {'Kp': [...], 'Ki': [...], 'Kd': [...]}.
        time_step (float): Time interval for PID updates.
        goal_position (list): Target position [X, Y, Z] in meters.
        max_iterations (int): Maximum steps to run the simulation.
        accuracy_threshold (float): Accuracy requirement for reaching the target.
        hold_duration (int): Consecutive steps the system must stay within the threshold.
        enable_render (bool): Flag to enable rendering the simulation environment.

    Returns:
        dict: Simulation results containing:
            - 'success' (bool): Whether the target was successfully reached.
            - 'steps_to_goal' (int): Steps required to reach the goal (if successful).
            - 'responses' (list): List of positions during the simulation.
            - 'time_steps' (list): List of time steps during the simulation.
    """
    simulation = Simulation(num_agents=1, render=enable_render)
    controller = PIDController(pid_gains['Kp'], pid_gains['Ki'], pid_gains['Kd'], time_step)

    state = simulation.reset(num_agents=1)
    agent_id = int(list(state.keys())[0].split('_')[-1])
    current_position = np.array(simulation.get_pipette_position(agent_id))

    controller.reset()
    in_threshold_counter = 0
    responses = []
    time_steps = []

    logging.info(f"Starting simulation with target position: {goal_position}")
    for iteration in range(max_iterations):
        control_signals = controller.compute(current_position, goal_position)
        action = np.clip(control_signals, -1, 1)
        action_with_dummy = np.concatenate([action, [0.0]])
        state = simulation.run([action_with_dummy])

        current_position = np.array(simulation.get_pipette_position(agent_id))
        responses.append(current_position)
        time_steps.append(iteration * time_step)

        distance_to_goal = np.linalg.norm(current_position - goal_position)
        logging.info(f"Step {iteration + 1}: Current Position: {current_position}, Distance to Goal: {distance_to_goal:.6f}")

        if distance_to_goal <= accuracy_threshold:
            in_threshold_counter += 1
            logging.debug(f"Within accuracy threshold for {in_threshold_counter} consecutive steps.")
            if in_threshold_counter >= hold_duration:
                logging.info(f"Goal reached successfully at step {iteration + 1}.")
                simulation.close()
                plot_response(time_steps, responses, goal_position, "PID Simulation Result")
                return {
                    "success": True,
                    "steps_to_goal": iteration + 1,
                    "responses": responses,
                    "time_steps": time_steps
                }
        else:
            in_threshold_counter = 0

    logging.warning("Maximum iterations reached. Goal not achieved.")
    simulation.close()
    plot_response(time_steps, responses, goal_position, "PID Simulation Result")
    return {
        "success": False,
        "steps_to_goal": None,
        "responses": responses,
        "time_steps": time_steps
    }

def execute_multiple_pid_tests(gain_values, time_step, test_count=1, hold_duration=50):
    """
    Executes multiple tests of the PID controller with random goal positions.

    Parameters:
        gain_values (dict): PID gain values {'Kp': [...], 'Ki': [...], 'Kd': [...]}.
        time_step (float): Time interval for PID updates.
        test_count (int): Number of tests to execute.
        hold_duration (int): Number of steps to confirm goal achievement.

    Logs:
        Number of successful tests and corresponding goal positions.
    """
    position_bounds = {
        "x": (-0.1872, 0.253),
        "y": (-0.1705, 0.2195),
        "z": (0.1693, 0.2895)
    }

    successes = 0
    logging.info("Starting multiple PID tests.")

    for test_index in range(test_count):
        goal_position = [
            np.random.uniform(*position_bounds["x"]),
            np.random.uniform(*position_bounds["y"]),
            np.random.uniform(*position_bounds["z"])
        ]

        logging.info(f"Test {test_index + 1}/{test_count}: Target position: {goal_position}")
        result = run_pid_simulation(
            pid_gains=gain_values,
            time_step=time_step,
            goal_position=goal_position,
            hold_duration=hold_duration
        )

        if result["success"]:
            logging.info(f"Test {test_index + 1}: Success! Steps to goal: {result['steps_to_goal']}")
            successes += 1
        else:
            logging.info(f"Test {test_index + 1}: Failure.")

    logging.info(f"PID Tests Complete: {successes}/{test_count} tests successful.")

if __name__ == "__main__":
    pid_gains = {
        "Kp": [15.0, 15.0, 15.0],
        "Ki": [0.0, 0.0, 0.0],
        "Kd": [0.8, 0.8, 0.8]
    }
    time_step = 1.0

    test_count = 1
    hold_duration = 50

    execute_multiple_pid_tests(
        gain_values=pid_gains,
        time_step=time_step,
        test_count=test_count,
        hold_duration=hold_duration
    )
