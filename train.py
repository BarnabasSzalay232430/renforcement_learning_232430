import os
import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

import wandb
from wandb.integration.sb3 import WandbCallback
from clearml import Task
from task10_ot2_gym_wrapper import OT2Env

# ----------------- ClearML and WandB Setup -----------------
task = Task.init(
    project_name="Mentor Group E/Group 3",
    task_name="OT2_RL_Training",
)
task.set_base_docker("deanis/2023y2b-rl:latest")
task.execute_remotely(queue_name="default")

def log_to_clearml(step, metric_name, value):
    task.get_logger().report_scalar(metric_name, "value", value=value, iteration=step)

# WandB initialization
os.environ['WANDB_API_KEY'] = 'cf5a05958641f64764dafe6badc9e911b54d9644'
run = wandb.init(project="ot2_digital_twin", sync_tensorboard=True)

# ----------------- Environment Setup -----------------
# Wrap the environment with VecNormalize for observation and reward normalization
env = OT2Env()

# Create dir to store models
model_dir = f"models/{run.id}"
os.makedirs(model_dir, exist_ok=True)
# ----------------- Argument Parsing -----------------
parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.0003, help="Learning rate for the PPO model")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for the PPO model")
parser.add_argument("--n_steps", type=int, default=2048, help="Number of steps per PPO update")
parser.add_argument("--n_epochs", type=int, default=10, help="Number of epochs for PPO optimization")
parser.add_argument("--timesteps", type=int, default=1_500_000)
args, _ = parser.parse_known_args()

def linear_schedule(initial_value):
    """
    Linear learning rate schedule.
    :param initial_value: (float) Initial learning rate.
    :return: (function) Schedule that computes the current learning rate depending on the remaining progress
    """
    def func(progress_remaining):
        # progress_remaining decreases from 1 (beginning) to 0 (end)
        return progress_remaining * initial_value
    return func


# ----------------- PPO Model Setup -----------------
# Define PPO
model = PPO(
    'MlpPolicy',
    env,
    verbose=1, 
    learning_rate=args.learning_rate, 
    batch_size=args.batch_size, 
    n_steps=args.n_steps, 
    n_epochs=args.n_epochs, 
    tensorboard_log=f"runs/{run.id}",
)


# ----------------- Callbacks -----------------
class CustomRewardLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomRewardLoggingCallback, self).__init__(verbose)
        self.rewards = []

    def _on_step(self) -> bool:
        if "rewards" in self.locals:
            reward = self.locals["rewards"][0]
            self.rewards.append(reward)
            avg_reward = np.mean(self.rewards[-100:])  # Rolling average of last 100 rewards
            wandb.log({"average_reward": avg_reward}, step=self.num_timesteps)
            log_to_clearml(self.num_timesteps, "average_reward", avg_reward)
        return True

    def _on_training_end(self) -> None:
        overall_avg_reward = np.mean(self.rewards)
        wandb.log({"final_average_reward": overall_avg_reward})

# Instantiate callbacks
wandb_callback = WandbCallback(
    model_save_freq=1000,
    model_save_path=model_dir,
    verbose=2
)
custom_reward_logging_callback = CustomRewardLoggingCallback()

# ----------------- Training Loop -----------------
time_steps_per_iter = 100000
num_iterations = 30

for iteration in range(1, num_iterations + 1):
    print(f"Starting iteration {iteration}")

    # Train the model
    model.learn(
        total_timesteps=time_steps_per_iter,
        callback=[wandb_callback, custom_reward_logging_callback],
        progress_bar=True,
        reset_num_timesteps=False,
        tb_log_name=f"run_{run.id}_iter_{iteration}",
    )

    # Save the model after each iteration
    model_path = f"{model_dir}/model_step_{time_steps_per_iter * iteration}"
    model.save(model_path)
    print(f"Model saved at iteration {iteration}: {model_path}")

# Final message
print("Training complete. Models and logs are saved.")
