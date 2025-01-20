import os
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from pid_class import PIDController
from sim_class import Simulation
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

print("Current working directory:", os.getcwd())

# Logging Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Constants
IMAGE_DIRECTORY = r"C:\Users\szala\Documents\GitHub\renforcement_learning_232430\textures\_plates"
COORDINATES_FILE = r"C:\Users\szala\Documents\GitHub\renforcement_learning_232430\roottip_coordinates.json"
START_POSITION = [0.10775, 0.062, 0.17]  # Initial pipette position in robot space
PID_GAINS = {"Kp": [15.0, 15.0, 15.0], "Ki": [0.0, 0.0, 0.0], "Kd": [0.8, 0.8, 0.8]}
TIME_STEP = 1.0
ACCURACY_THRESHOLD = 0.001
HOLD_DURATION = 50
MAX_ITERATIONS = 1000

# Load root tip coordinates from the JSON file
def load_coordinates(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Plot the PID response
def plot_response(time_steps, responses, goal_position, title):
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

# Run PID simulation for a single coordinate
def run_pid_simulation(simulation, pid_gains, time_step, goal_position, max_iterations, accuracy_threshold, hold_duration):
    controller = PIDController(pid_gains['Kp'], pid_gains['Ki'], pid_gains['Kd'], time_step)

    # Reset simulation and set the start position
    state = simulation.reset(num_agents=1)
    agent_id = int(list(state.keys())[0].split('_')[-1])
    simulation.set_start_position(*START_POSITION)

    # Initialize PID controller and tracking
    controller.reset()
    in_threshold_counter = 0
    responses = []
    time_steps = []
    current_position = np.array(simulation.get_pipette_position(agent_id))

    logging.info(f"Starting PID control to reach goal position: {goal_position}")

    for iteration in range(max_iterations):
        control_signals = controller.compute(current_position, goal_position)
        action = np.clip(control_signals, -1, 1)
        action_with_dummy = np.concatenate([action, [0.0]])
        simulation.run([action_with_dummy])

        current_position = np.array(simulation.get_pipette_position(agent_id))
        responses.append(current_position)
        time_steps.append(iteration * time_step)

        distance_to_goal = np.linalg.norm(current_position - goal_position)
        logging.info(f"Iteration {iteration + 1}: Current Position: {current_position}, Distance to Goal: {distance_to_goal:.6f}")

        if distance_to_goal <= accuracy_threshold:
            in_threshold_counter += 1
            if in_threshold_counter >= hold_duration:
                logging.info(f"Goal position reached successfully at iteration {iteration + 1}.")
                plot_response(time_steps, responses, goal_position, "PID Simulation Result")
                return {
                    "success": True,
                    "steps_to_goal": iteration + 1,
                    "responses": responses,
                    "time_steps": time_steps,
                }
        else:
            in_threshold_counter = 0

    logging.warning(f"Maximum iterations reached. Goal position not achieved.")
    plot_response(time_steps, responses, goal_position, "PID Simulation Result")
    return {
        "success": False,
        "steps_to_goal": None,
        "responses": responses,
        "time_steps": time_steps,
    }

# Main simulation pipeline
def main_simulation(image_directory, coordinates_file):
    # Load coordinates
    root_coordinates = load_coordinates(coordinates_file)

    # Initialize simulation
    simulation = Simulation(num_agents=1, render=True)

    try:
        for image_name, coordinates in root_coordinates.items():
            logging.info(f"Processing image: {image_name}")

            # Ensure the image exists in the directory
            image_path = os.path.join(image_directory, image_name)
            if not os.path.exists(image_path):
                logging.warning(f"Image not found: {image_path}")
                continue

            # Process each coordinate
            for idx, coordinate in enumerate(coordinates):
                logging.info(f"Starting PID control for root tip {idx + 1} in image {image_name}.")
                result = run_pid_simulation(
                    simulation=simulation,
                    pid_gains=PID_GAINS,
                    time_step=TIME_STEP,
                    goal_position=coordinate,
                    max_iterations=MAX_ITERATIONS,
                    accuracy_threshold=ACCURACY_THRESHOLD,
                    hold_duration=HOLD_DURATION,
                )

                if result["success"]:
                    logging.info(f"Successfully reached root tip {idx + 1} in image {image_name}.")
                else:
                    logging.error(f"Failed to reach root tip {idx + 1} in image {image_name}.")
    finally:
        simulation.close()
        logging.info("Simulation completed.")

# Main Execution
if __name__ == "__main__":
    main_simulation(IMAGE_DIRECTORY, COORDINATES_FILE)
