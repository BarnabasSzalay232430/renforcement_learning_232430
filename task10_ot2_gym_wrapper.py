import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Change the current working directory to the script's directory
os.chdir(script_dir)

print("Current working directory:", os.getcwd())


class OT2Env(gym.Env):
    def __init__(self, render=False, max_steps=1000):
        super(OT2Env, self).__init__()
        self.render = render
        self.max_steps = max_steps

        # Create the simulation environment
        self.sim = Simulation(num_agents=1)

        # Define action space (velocity for x, y, z)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # Define observation space (pipette position + goal position)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        # Initialize variables
        self.steps = 0
        self.goal_position = None

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        # Reset the simulation
        self.sim.reset(num_agents=1)

        # Randomly initialize the goal position within a predefined range
        self.goal_position = np.random.uniform(-0.3, 0.3, size=(3,)).astype(np.float32)

        # Extract the pipette position dynamically
        state = self.sim.get_states()
        print("Simulation state:", state)  # Debug print to inspect structure
        robot_id = list(state.keys())[0]  # Dynamically fetch the first robot ID
        pipette_position = np.array(state[robot_id]['pipette_position'], dtype=np.float32)

        # Observation includes pipette position and goal position
        observation = np.concatenate([pipette_position, self.goal_position])

        self.steps = 0

        # Return observation and an empty info dictionary
        return observation, {}


    def step(self, action):
        # Append a dummy action for the simulator (e.g., dispensing liquid)
        sim_action = np.append(action, 0.0)

        # Apply the action in the simulation
        self.sim.run([sim_action])

        # Get updated state dynamically
        state = self.sim.get_states()
        robot_id = list(state.keys())[0]  # Dynamically fetch the first robot ID
        pipette_position = np.array(state[robot_id]['pipette_position'], dtype=np.float32)
        observation = np.concatenate([pipette_position, self.goal_position])

        # Calculate reward (negative distance to goal)
        distance_to_goal = np.linalg.norm(pipette_position - self.goal_position)
        reward = float(-distance_to_goal)  # Convert reward to native Python float

        # Check termination conditions
        terminated = bool(distance_to_goal < 0.01)  # Ensure terminated is a Python boolean
        truncated = bool(self.steps >= self.max_steps)  # Ensure truncated is a Python boolean

        # Update step count
        self.steps += 1

        # Return Gym-compliant step information
        info = {}
        return observation, reward, terminated, truncated, info



    def render(self, mode='human'):
        if self.render:
            print(f"Rendering at step {self.steps}. Current pipette position: {self.sim.get_state()['robotId_1']['pipette_position']}")

    def close(self):
        self.sim.close()