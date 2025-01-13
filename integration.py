import os
import logging
from sim_class import Simulation  # Import the Simulation class

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("simulation_log.txt"), logging.StreamHandler()],
)

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Initialize the simulation
simulation = Simulation(num_agents=1, render=True)

# Path to the image file
image_path = os.path.join(r"C:\\Users\\szala\\Documents\\GitHub\\renforcement_learning_232430\\kaggle_test_extracted\\test_image_3.png")  # Update the image filename as needed

# Ensure the image file exists
if not os.path.exists(image_path):
    logging.error(f"Image file not found: {image_path}")
    simulation.close()
    exit(1)

# Load the image onto the plate
simulation.load_plate_image(image_path)

# Run the simulation to visualize the plate with the image
logging.info("Running the simulation...")
try:
    simulation.run([], num_steps=240)  # Run for 240 steps
finally:
    # Close the simulation
    simulation.close()
    logging.info("Simulation closed.")
