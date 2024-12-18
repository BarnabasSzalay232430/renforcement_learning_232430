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
env = DummyVecEnv([lambda: OT2Env(render=False)])
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

# ----------------- Argument Parsing -----------------
parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.0003, help="Learning rate for the PPO model")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for the PPO model")
parser.add_argument("--n_steps", type=int, default=2048, help="Number of steps per PPO update")
parser.add_argument("--n_epochs", type=int, default=10, help="Number of epochs for PPO optimization")
parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for rewards")
parser.add_argument("--clip_range", type=float, default=0.2, help="Clipping range for PPO")
args, _ = parser.parse_known_args()

# ----------------- PPO Model Setup -----------------
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=args.learning_rate,
    batch_size=args.batch_size,
    n_steps=args.n_steps,
    n_epochs=args.n_epochs,
    gamma=args.gamma,
    clip_range=args.clip_range,
    tensorboard_log=f"runs/{run.id}",
)

# Directory for saving models
model_dir = f"models/{run.id}"
os.makedirs(model_dir, exist_ok=True)

# ----------------- Callbacks -----------------
class CustomWandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomWandbCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        # Check if 'infos' exists in self.locals
        if 'infos' in self.locals and len(self.locals['infos']) > 0:
            episode_info = self.locals['infos'][0].get('episode', {})  # Safely extract episode info

            # Log episode rewards
            if 'r' in episode_info:  # Episode reward
                self.episode_rewards.append(episode_info['r'])
                mean_reward = np.mean(self.episode_rewards[-100:])  # Mean of the last 100 rewards
                wandb.log({"mean_reward": mean_reward}, step=self.num_timesteps)
                log_to_clearml(self.num_timesteps, "mean_reward", mean_reward)

        return True

    def _on_training_end(self) -> None:
        # Log aggregate statistics at the end of training
        wandb.log({
            "average_episode_reward": np.mean(self.episode_rewards),
            "average_episode_length": np.mean(self.episode_lengths),
        })

# Instantiate callbacks
wandb_callback = WandbCallback(
    model_save_freq=1000,
    model_save_path=model_dir,
    verbose=2
)
custom_wandb_callback = CustomWandbCallback()

# ----------------- Training Loop -----------------
time_steps_per_iter = 100000
num_iterations = 10

for iteration in range(1, num_iterations + 1):
    print(f"Starting iteration {iteration}")
    
    # Train the model
    model.learn(
        total_timesteps=time_steps_per_iter,
        callback=[wandb_callback, custom_wandb_callback],
        progress_bar=True,
        reset_num_timesteps=False,
        tb_log_name=f"runs/{run.id}_iter_{iteration}",
    )

    # Save the model after each iteration
    model_path = f"{model_dir}/model_step_{time_steps_per_iter * iteration}"
    model.save(model_path)
    print(f"Model saved at iteration {iteration}: {model_path}")