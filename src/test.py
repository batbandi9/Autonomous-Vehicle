#!/usr/bin/env python3
"""
Load a trained PPO model and watch it perform in the environment.
"""

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
import time
import os # <-- Added the import os here

# Import your environment class
from env import ParkingPPOEnv # 

# Path to the saved model
MODEL_PATH = "ppo_logs19/ppo_parking_final.zip"
#MODEL_PATH = "ppo_curriculum_logs_5/final_extended_model.zip"

def test_model():
    # 1. Create the environment
    # We use render_mode="human" to see the simulation
    env = ParkingPPOEnv(render_mode="human") # <--- CORRECTED CLASS NAME
    
    # Optional: If you want to see episode stats, you can wrap it
    # from gymnasium.wrappers import RecordEpisodeStatistics
    # env = RecordEpisodeStatistics(env)

    # 2. Load the trained model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return
        
    print(f"Loading model from {MODEL_PATH}...")
    model = PPO.load(MODEL_PATH)
    print("Model loaded.")

    # 3. Run the test loop
    num_episodes = 10
    max_steps = 1000
    for ep in range(num_episodes):
        obs, info = env.reset()
        terminated = False
        truncated = False
        episode_reward = 0
        step = 0
        print(f"\n--- Starting Episode {ep + 1}/{num_episodes} ---")
        while not (terminated or truncated) and step < max_steps:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step += 1
            time.sleep(0.01)
        print(f"Episode {ep + 1} finished after {step} steps.")
        print(f"Episode Reward: {episode_reward:.2f}")
        if info.get('is_success'):
            print("Status: PARKED SUCCESSFULLY!")
        else:
            print("Status: Failed (crashed or timed out).")

    # 4. Close the environment
    env.close()
    print("\nTesting complete.")

if __name__ == "__main__":
    test_model()