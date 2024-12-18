import os
import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import wandb
from wandb.integration.sb3 import WandbCallback
from clearml import Task
from task10_ot2_gym_wrapper import OT2Env

# ----------------- ClearML and WandB Setup -----------------
# ClearML task initialization
task = Task.init(
    project_name="Mentor Group E/Group 3",
    task_name="OT2_RL_Training",
)
task.set_base_docker("deanis/2023y2b-rl:latest")
task.execute_remotely(queue_name="default")

# WandB initialization
os.environ['WANDB_API_KEY'] = 'cf5a05958641f64764dafe6badc9e911b54d9644'
run = wandb.init(project="ot2_digital_twin", sync_tensorboard=True)

# ----------------- Environment Setup -----------------
env = OT2Env()  # Assuming `render` is a parameter in OT2Env

# ----------------- Argument Parsing -----------------
parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.0003, help="Learning rate for the PPO model")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for the PPO model")
parser.add_argument("--n_steps", type=int, default=2048, help="Number of steps per PPO update")
parser.add_argument("--n_epochs", type=int, default=10, help="Number of epochs for PPO optimization")
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
        self.success_rate = []

    def _on_step(self) -> bool:
        # Collect episode rewards and lengths
        if 'episode' in self.locals:
            episode_info = self.locals['infos'][0].get('episode', {})
            if 'r' in episode_info:  # Episode reward
                self.episode_rewards.append(episode_info['r'])
                wandb.log({"episode_reward": episode_info['r']}, step=self.num_timesteps)
            if 'l' in episode_info:  # Episode length
                self.episode_lengths.append(episode_info['l'])
                wandb.log({"episode_length": episode_info['l']}, step=self.num_timesteps)

        # Success rate logging (if available in env info)
        success = self.locals['infos'][0].get('success', None)
        if success is not None:
            self.success_rate.append(success)
            wandb.log({"success_rate": np.mean(self.success_rate)}, step=self.num_timesteps)

        # Log entropy (policy exploration)
        entropy = self.model.logger.name_to_value.get('entropy', None)
        if entropy is not None:
            wandb.log({"entropy": entropy}, step=self.num_timesteps)

        # Log learning rate
        wandb.log({"learning_rate": self.model.learning_rate}, step=self.num_timesteps)

        return True

    def _on_training_end(self) -> None:
        # Log aggregate statistics at the end of training
        wandb.log({
            "average_episode_reward": np.mean(self.episode_rewards),
            "average_episode_length": np.mean(self.episode_lengths),
            "success_rate": np.mean(self.success_rate)
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
        tb_log_name=f"runs/{run.id}_iter_{iteration}"
    )

    # Save the model after each iteration
    model_path = f"{model_dir}/model_step_{time_steps_per_iter * iteration}"
    model.save(model_path)
    print(f"Model saved at iteration {iteration}: {model_path}")

# Final message
print("Training complete. Models and logs are saved.")
