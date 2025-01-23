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
    """
    OpenAI Gym-compatible environment for controlling an OT2 robot in simulation.
    
    Attributes:
        initial_position (np.array): Starting position of the robot's pipette.
        render (bool): Flag to enable or disable rendering.
        max_steps (int): Maximum steps allowed in an episode.
        normalize_rewards (bool): Whether to normalize rewards to [0, 1].
        sim (Simulation): Simulation environment instance.
        action_space (gym.spaces.Box): Defines the range of valid actions.
        observation_space (gym.spaces.Box): Defines the range of valid observations.
        steps (int): Step counter for the current episode.
        cumulative_reward (float): Total reward accumulated in the current episode.
        goal_position (np.array): Target position the robot needs to reach.
    """
    
    def __init__(self, initial_position=np.array([0.10775, 0.062, 0.12]), render=False, max_steps=1000, normalize_rewards=True):
        """
        Initialize the OT2 environment.

        Args:
            initial_position (np.array): Starting position of the pipette (default: [0.10775, 0.062, 0.12]).
            render (bool): Enable or disable rendering (default: False).
            max_steps (int): Maximum steps allowed in an episode (default: 1000).
            normalize_rewards (bool): Normalize rewards to [0, 1] (default: True).
        """
        self.initial_position = initial_position
        self.render = render
        self.max_steps = max_steps
        self.normalize_rewards = normalize_rewards

        # Initialize the simulation environment
        self.sim = Simulation(num_agents=1)

        # Define action space: 3 continuous values (x, y, z control)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # Define observation space: 6 continuous values (pipette position + goal position)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        # Initialize internal variables
        self.steps = 0
        self.cumulative_reward = 0.0
        self.goal_position = None

    def reset(self, seed=None):
        """
        Reset the environment to its initial state.

        Args:
            seed (int, optional): Seed for reproducibility (default: None).

        Returns:
            tuple: Initial observation and an empty info dictionary.
        """
        if seed is not None:
            np.random.seed(seed)

        # Reset the simulation environment
        self.sim.reset(num_agents=1)
        self.sim.set_start_position(self.initial_position[0], self.initial_position[1], self.initial_position[2])

        # Generate a random goal position within a defined range
        self.goal_position = np.random.uniform(-0.3, 0.3, size=(3,)).astype(np.float32)

        # Get the initial pipette position and combine it with the goal position for observation
        state = self.sim.get_states()
        robot_id = list(state.keys())[0]
        pipette_position = np.array(state[robot_id].get('pipette_position', [0, 0, 0]), dtype=np.float32)
        observation = np.concatenate([pipette_position, self.goal_position])

        # Reset counters
        self.steps = 0
        self.cumulative_reward = 0.0

        return observation, {}

    def step(self, action):
        """
        Execute a single time step in the environment.

        Args:
            action (np.array): Action to be performed (x, y, z control).

        Returns:
            tuple: Observation, reward, terminated flag, truncated flag, and info dictionary.
        """
        # Clip the action values to the valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Run the simulation with the action
        sim_action = np.append(action, 0.0)  # Append unused action dimension
        self.sim.run([sim_action])

        # Get the current pipette position and construct the observation
        state = self.sim.get_states()
        robot_id = list(state.keys())[0]
        pipette_position = np.array(state[robot_id].get('pipette_position', [0, 0, 0]), dtype=np.float32)
        observation = np.concatenate([pipette_position, self.goal_position])

        # Calculate the distance to the goal and compute the reward
        distance_to_goal = np.linalg.norm(pipette_position - self.goal_position)
        max_distance = np.sqrt(3) * 0.6  # Approximate max distance in the range
        reward = -distance_to_goal / max_distance
        if self.normalize_rewards:
            reward = (reward + 1) / 2  # Normalize to [0, 1]

        # Update cumulative reward
        self.cumulative_reward += reward

        # Check termination and truncation conditions
        terminated = distance_to_goal < 0.01  # Success condition
        truncated = self.steps >= self.max_steps  # Maximum steps condition

        # Increment step counter
        self.steps += 1

        # Additional information for debugging or evaluation
        info = {
            "distance_to_goal": distance_to_goal,
            "average_reward": self.cumulative_reward / self.steps
        }

        return observation, reward, terminated, truncated, info

    def get_plate_image(self):
        """
        Retrieve the path of the plate image from the simulation.

        Returns:
            str: Path to the plate image.
        """
        return self.sim.get_plate_image()

    def render(self, mode='human'):
        """
        Render the current state of the environment.

        Args:
            mode (str): Rendering mode (default: 'human').
        """
        if self.render:
            state = self.sim.get_states()
            robot_id = list(state.keys())[0]
            pipette_position = state[robot_id]['pipette_position']
            print(f"Rendering at step {self.steps}. Current pipette position: {pipette_position}")

    def close(self):
        """
        Close the environment and clean up resources.
        """
        self.sim.close()
