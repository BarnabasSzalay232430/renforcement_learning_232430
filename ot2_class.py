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
    def __init__(self, initial_position=np.array([0.10775, 0.062, 0.12]), render=False, max_steps=1000, normalize_rewards=True):
        self.initial_position = initial_position

        super(OT2Env, self).__init__()
        self.render = render
        self.max_steps = max_steps
        self.normalize_rewards = normalize_rewards

        # Create the simulation environment
        self.sim = Simulation(num_agents=1)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        
        # Initialize variables
        self.steps = 0
        self.cumulative_reward = 0.0
        self.goal_position = None

    # In OT2Env class (reset method):
    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        # Reset the simulation
        self.sim.reset(num_agents=1)
        
        # Set the robot's starting position using the correct method
        self.sim.set_start_position(self.initial_position[0], self.initial_position[1], self.initial_position[2])  # Set start position

        # Set the goal position randomly or from a predefined source
        self.goal_position = np.random.uniform(-0.3, 0.3, size=(3,)).astype(np.float32)

        # Get the initial state
        state = self.sim.get_states()
        robot_id = list(state.keys())[0]
        pipette_position = np.array(state[robot_id].get('pipette_position', [0, 0, 0]), dtype=np.float32)
        observation = np.concatenate([pipette_position, self.goal_position])
        self.steps = 0
        self.cumulative_reward = 0.0
        return observation, {}

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        sim_action = np.append(action, 0.0)
        self.sim.run([sim_action])
        
        state = self.sim.get_states()
        robot_id = list(state.keys())[0]
        pipette_position = np.array(state[robot_id].get('pipette_position', [0, 0, 0]), dtype=np.float32)
        observation = np.concatenate([pipette_position, self.goal_position])

        distance_to_goal = np.linalg.norm(pipette_position - self.goal_position)
        max_distance = np.sqrt(3) * 0.6
        reward = -distance_to_goal / max_distance
        if self.normalize_rewards:
            reward = (reward + 1) / 2  # Normalize to [0, 1]

        self.cumulative_reward += reward
        terminated = distance_to_goal < 0.01
        truncated = self.steps >= self.max_steps
        self.steps += 1

        info = {"distance_to_goal": distance_to_goal, "average_reward": self.cumulative_reward / self.steps}
        return observation, reward, terminated, truncated, info

    def get_plate_image(self):
        """
        Proxy method to get the plate image path from the Simulation class.
        """
        return self.sim.get_plate_image()

    def render(self, mode='human'):
        if self.render:
            state = self.sim.get_states()
            robot_id = list(state.keys())[0]
            pipette_position = state[robot_id]['pipette_position']
            print(f"Rendering at step {self.steps}. Current pipette position: {pipette_position}")

    def close(self):
        self.sim.close()
