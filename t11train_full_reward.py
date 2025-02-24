import time
import numpy as np
from stable_baselines3 import PPO
from t10envw_full_reward import OT2Env
import os
import wandb
from wandb.integration.sb3 import WandbCallback
import argparse
from clearml import Task
import sys
import gymnasium as gym
import tensorflow
from typing_extensions import TypeIs

# Env
env = OT2Env()

# ClearML
task = Task.init(project_name="Mentor Group E/Group 3", task_name="Final?")
task.set_base_docker('deanis/2023y2b-rl:latest')
task.execute_remotely(queue_name="default")

# WandB
os.environ['WANDB_API_KEY'] = 'cf5a05958641f64764dafe6badc9e911b54d9644'
run = wandb.init(project="RL_train_more_reward",sync_tensorboard=True)

# Create dir to store models
model_dir = f"models/{run.id}"
os.makedirs(model_dir, exist_ok=True)

# Arg parsing
parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.0001)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--n_steps", type=int, default=2048)
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--timesteps", type=int, default=1_500_000)
args = parser.parse_args()

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


# Classic Callback
wandb_callback = WandbCallback(
    model_save_freq = 100_000,
    model_save_path = f"models/{run.id}",
    verbose = 2
)

# Train
model.learn(
    total_timesteps=args.timesteps,
    callback=wandb_callback,
    reset_num_timesteps=False,
    progress_bar=True,
    tb_log_name=f"runs/{run.id}"
)
model.save(f"models/{run.id}/{args.timesteps}_baseline")
wandb.save(f"models/{run.id}/{args.timesteps}_baseline")