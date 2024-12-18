from stable_baselines3 import PPO
import gymnasium as gym
import os
import wandb
from wandb.integration.sb3 import WandbCallback
from clearml import Task
import argparse

# Import your custom OT2 Environment
from task10_ot2_gym_wrapper import OT2Env

# ----------------- ClearML Setup -----------------
# Initialize ClearML Task
task = Task.init(project_name='Mentor Group E/Group 3', task_name='OT2_RL_Training')
task.set_base_docker('deanis/2023y2b-rl:latest')  # Docker image for remote execution
task.execute_remotely(queue_name="default")

# ----------------- WandB Setup -----------------
os.environ['WANDB_API_KEY'] = 'cf5a05958641f64764dafe6badc9e911b54d9644'  # Replace with your WandB API key
run = wandb.init(project="ot2_digital_twin", sync_tensorboard=True)

# ----------------- Hyperparameter Setup -----------------
parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.0003)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--n_steps", type=int, default=2048)
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--total_timesteps", type=int, default=1000000)
args = parser.parse_args()

# ----------------- Environment Setup -----------------
env = OT2Env(render=False, max_steps=1000)  # Instantiate your OT2 environment

# ----------------- PPO Model Setup -----------------
model = PPO(
    'MlpPolicy',
    env,
    verbose=1,
    learning_rate=args.learning_rate,
    batch_size=args.batch_size,
    n_steps=args.n_steps,
    n_epochs=args.n_epochs,
    tensorboard_log=f"runs/{run.id}"
)

# ----------------- WandB Callback -----------------
wandb_callback = WandbCallback(
    model_save_freq=10000,
    model_save_path=f"models/{run.id}",
    verbose=2
)

# ----------------- Training Loop -----------------
time_steps_per_iter = 100000
for i in range(args.total_timesteps // time_steps_per_iter):
    model.learn(
        total_timesteps=time_steps_per_iter,
        callback=wandb_callback,
        reset_num_timesteps=False,
        tb_log_name=f"runs/{run.id}"
    )
    # Save the model incrementally
    model.save(f"models/{run.id}/model_step_{(i+1)*time_steps_per_iter}")

print("Training Complete! Model Saved.")
