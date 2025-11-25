#!/usr/bin/env python3
"""
A simpler PPO CURRICULUM LEARNING script.

*** MODIFIED TO CONTINUE TRAINING FROM A FINAL MODEL ***
This script will:
1. Load 'ppo_curriculum_logs_4/final_curriculum_model.zip'
2. Train 1,000,000 more steps with threshold 1.0
3. Train 1,000,000 more steps with threshold 0.5
"""

import os
import gymnasium as gym
import numpy as np
# === IMPORT THE WRAPPER ===
from gymnasium.wrappers import RecordEpisodeStatistics
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.logger import configure

# Import your NEW PPO-compatible environment
from env import ParkingPPOEnv 

# Training parameters
# <=== NEW LOG DIRECTORY ===>
LOG_DIR = "ppo_curriculum_logs_5/" 
N_ENVS = 8  # Parallel environments

# <=== YOUR NEW CURRICULUM ===>
# These are the ONLY stages that will run
CURRICULUM = [
    (1.0, 1_000_000),   # Stage 1 (Total 9M)
    (0.5, 1_000_000),   # Stage 2 (Total 10M)
]

# <=== CONFIGURATION TO CONTINUE TRAINING ===>
# -------------------------------------------------------------------------
# Set this to the path of your *final* model
LOAD_MODEL_PATH = "ppo_curriculum_logs_4/final_curriculum_model.zip" 

# Set this to 0 to start from the beginning of the NEW curriculum list above
START_STAGE_INDEX = 0
# -------------------------------------------------------------------------


def make_env(threshold, rank):
    """Create environment with specific success threshold"""
    def _init():
        env = ParkingPPOEnv(render_mode=None) 
        env.unwrapped.success_cost_threshold = threshold
        
        # === ADD THIS WRAPPER ===
        # This wrapper tracks episode stats (reward/length)
        env = RecordEpisodeStatistics(env)
        
        return env
    return _init


def train_ppo_curriculum():
    """Train PPO agent with curriculum learning"""
    print("PPO CURRICULUM LEARNING (Simple)")
    print("\nContinuing Training with New Stages:")
    for i, (threshold, timesteps) in enumerate(CURRICULUM, 1):
        print(f"  Stage {i}: threshold={threshold:.1f}, train for {timesteps:,} steps")
    
    os.makedirs(LOG_DIR, exist_ok=True)
    
    model = None
    VecEnv = SubprocVecEnv 
    
    for stage_idx, (threshold, timesteps) in enumerate(CURRICULUM):
        
        if stage_idx < START_STAGE_INDEX:
            print(f"\nSKIPPING Stage {stage_idx + 1} (already completed).")
            continue
        
        # We add "+5" to the stage name just to show it's a new run
        stage_name = f"Stage_{stage_idx + 5}_threshold_{threshold:.1f}"
        stage_log_path = LOG_DIR + stage_name
        
        print(f"\n🎯 STARTING: {stage_name}")
        
        train_env = VecEnv([make_env(threshold, i) for i in range(N_ENVS)])
        
        if model is None:
            # This block runs ONLY for the FIRST stage we are training
            
            # <=== MODIFIED LOADING LOGIC ===>
            # Load if LOAD_MODEL_PATH is set AND we are on the starting stage
            if LOAD_MODEL_PATH is not None and stage_idx == START_STAGE_INDEX:
                # We are RESUMING or CONTINUING training
                print(f"Loading model to continue training! Path: {LOAD_MODEL_PATH}")
                
                if not os.path.exists(LOAD_MODEL_PATH):
                    raise ValueError(f"Error: LOAD_MODEL_PATH file not found: {LOAD_MODEL_PATH}")
                
                # Load the model and pass the new environment
                model = PPO.load(LOAD_MODEL_PATH, env=train_env, custom_objects={'learning_rate': 3e-4}) # Make sure to set any custom params if needed
                
                # Set up the new logger for this resumed stage
                print(f"Setting new logger to: {stage_log_path}")
                new_logger = configure(stage_log_path, ["stdout", "tensorboard"])
                model.set_logger(new_logger)
            
            else:
                # We are starting a NEW training run from scratch
                print("Creating new PPO model...")
                model = PPO(
                    "MlpPolicy",
                    train_env,
                    learning_rate=3e-4,
                    n_steps=2048,
                    batch_size=64,
                    n_epochs=10,
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_range=0.2,
                    ent_coef=0.01, 
                    vf_coef=0.5,
                    max_grad_norm=0.5,
                    policy_kwargs=dict(
                        net_arch=dict(pi=[256, 256], vf=[256, 256])
                    ),
                    verbose=1, # This will print PPO logs
                    tensorboard_log=stage_log_path 
                )
        else:
            # This block runs for subsequent stages
            print(f"Continuing model from previous stage...")
            model.set_env(train_env)
            print(f"Setting new logger to: {stage_log_path}")
            new_logger = configure(stage_log_path, ["stdout", "tensorboard"])
            model.set_logger(new_logger)
        
        # We still use CheckpointCallback to save our work
        checkpoint_callback = CheckpointCallback(
            save_freq=max(10_000 // N_ENVS, 1),
            save_path=stage_log_path + "/",
            name_prefix=f"checkpoint_{stage_name}",
            verbose=1 # This will print when it saves
        )
        
        print(f"\n🚀 Training for {timesteps:,} timesteps...")
        try:
            model.learn(
                total_timesteps=timesteps,
                callback=checkpoint_callback,
                reset_num_timesteps=False, # <-- This is CRITICAL for resuming
                progress_bar=True 
            )
        except KeyboardInterrupt:
            print(f"\n⚠️  Training interrupted during {stage_name}!")
            # Save the model on interrupt
            interrupt_save_path = f"{stage_log_path}/interrupted_model.zip"
            model.save(interrupt_save_path)
            print(f"Model saved to {interrupt_save_path}")
            break
        
        stage_save_path = f"{LOG_DIR}{stage_name}_final.zip"
        model.save(stage_save_path)
        print(f"✅ {stage_name} complete! Model saved to {stage_save_path}")
        
        train_env.close()
    
    print(f"\n🎉 EXTENDED CURRICULUM COMPLETE!")
    
    if model is not None:
        final_path = f"{LOG_DIR}final_extended_model.zip"
        model.save(final_path)
        print(f"Final model saved to: {final_path}")
    else:
        print("Model was not trained.")


if __name__ == "__main__":
    train_ppo_curriculum()