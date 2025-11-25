#!/usr/bin/env python3

import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.logger import configure
from gymnasium.wrappers import RecordEpisodeStatistics

from env import ParkingPPOEnv  # Your PPO-compatible environment

# Training parameters
LOG_DIR = "ppo_logs19/"  # TensorBoard logs folder
N_ENVS = 8  # Number of parallel environments
TOTAL_TIMESTEPS = 5_000_000  # 3 million steps

# Helper function to create environments
def make_env(rank):
    def _init():
        env = ParkingPPOEnv(render_mode=None)
        # Optional: set your success threshold if you want (2m + 2deg)
        env.unwrapped.success_cost_threshold = 4e-4
        env = RecordEpisodeStatistics(env)
        return env
    return _init

if __name__ == "__main__":
    os.makedirs(LOG_DIR, exist_ok=True)

    # Create vectorized environments for faster training
    train_env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])

    # Configure logger (stdout + tensorboard)
    logger = configure(LOG_DIR, ["stdout", "tensorboard"])

    # Create PPO model with default parameters
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log=LOG_DIR
    )
    model.set_logger(logger)

    print("🚀 Starting PPO training (simple, default params, 4M steps)...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    # Save the final trained model
    final_model_path = os.path.join(LOG_DIR, "ppo_parking_final.zip")
    model.save(final_model_path)
    print(f" Training complete! Model saved to: {final_model_path}")

    # Close environment
    train_env.close()
