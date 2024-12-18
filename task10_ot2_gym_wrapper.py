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
        # Use a random generator for isolated random states
        rng = np.random.default_rng(seed)

        # Reset the simulation
        self.sim.reset(num_agents=1)

        # Randomly initialize the goal position within a predefined range
        self.goal_position = rng.uniform(-0.3, 0.3, size=(3,)).astype(np.float32)

        # Extract the pipette position dynamically
        state = self.sim.get_states()
        if not state:
            raise RuntimeError("Simulation state is empty or invalid!")
        robot_id = list(state.keys())[0]  # Dynamically fetch the first robot ID
        pipette_position = np.array(state[robot_id].get('pipette_position', [0, 0, 0]), dtype=np.float32)

        # Observation includes pipette position and goal position
        observation = np.concatenate([pipette_position, self.goal_position])

        # Check observation validity
        assert self.observation_space.contains(observation), "Observation is out of bound"

        self.steps = 0

        # Return observation and an empty info dictionary
        return observation, {}

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        sim_action = np.append(action, 0.0)
        self.sim.run([sim_action])

        state = self.sim.get_states()
        if not state:
            raise RuntimeError("Simulation state is empty or invalid!")
        robot_id = list(state.keys())[0]
        pipette_position = np.array(state[robot_id].get('pipette_position', [0, 0, 0]), dtype=np.float32)
        observation = np.concatenate([pipette_position, self.goal_position])

        distance_to_goal = np.linalg.norm(pipette_position - self.goal_position)
        max_distance = np.sqrt(3) * 0.6
        reward = float(-distance_to_goal / max_distance)

        # Reward shaping: give positive rewards for progress
        progress_reward = 1 - (distance_to_goal / max_distance)
        reward += progress_reward

        terminated = bool(distance_to_goal < 0.01)
        truncated = bool(self.steps >= self.max_steps)

        # Log success for debugging
        if terminated:
            print(f"Success: Reached goal at step {self.steps}. Distance: {distance_to_goal}")

        self.steps += 1
        return observation, reward, terminated, truncated, {}


    def render(self, mode='human'):
        if self.render:
            state = self.sim.get_states()
            robot_id = list(state.keys())[0]
            pipette_position = state[robot_id]['pipette_position']
            print(f"Rendering at step {self.steps}. Current pipette position: {pipette_position}")

    def close(self):
        self.sim.close()
