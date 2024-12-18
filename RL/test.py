from sim_class import Simulation
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Change the current working directory to the script's directory
os.chdir(script_dir)

print("Current working directory:", os.getcwd())

# Initialize the simulation with a specified number of agents
sim = Simulation(num_agents=1)  # For one robot

# Define the actions to move to each corner of the working envelope
corners = [
    [1.0, 1.0, 1.0, 0],  # Top-front-right corner
    [-1.0, 0, 0, 0],  # Top-front-left corner
    [0, -1.0, 0, 0],  # Top-back-right corner
    [1.0, 0, 0, 0],  # Top-back-left corner
    [0, 0, -1.0, 0],  # Bottom-front-right corner
    [0, 1.0, 0, 0],  # Bottom-front-left corner
    [-1.0, 0, 0, 0],  # Bottom-back-right corner
    [0, -1.0, 0, 0]  # Bottom-back-left corner
]

# Move to each corner and print only the pipette position
for idx, corner_action in enumerate(corners):
    print(f"Moving to corner {idx + 1}: {corner_action}")
    state = sim.run([corner_action], num_steps=160)
    pipette_position = state['robotId_1']['pipette_position']  # Extract pipette position
    print(f"Pipette position at corner {idx + 1}: {pipette_position}")
