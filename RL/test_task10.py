from stable_baselines3.common.env_checker import check_env
from task10_ot2_gym_wrapper import OT2Env
import numpy as np

# Create an instance of the custom environment
env = OT2Env(render=False)

# Check the environment's compatibility
check_env(env)

# Run a test with random actions
num_episodes = 1

for episode in range(num_episodes):
    obs = env.reset()
    done = False
    step = 0

    print(f"\n--- Starting Episode {episode + 1} ---")
    while not done:
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step: {step + 1}, Action: {action}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
        step += 1
        if terminated or truncated:
            done = True
            print(f"Episode finished after {step} steps.")
env.close()
