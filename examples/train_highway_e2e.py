"""
Example: Training an E2E agent for highway driving.

This script demonstrates how to train an end-to-end RL agent
for general highway driving.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from autonomous_vehicle.environments import HighwayEnv
from autonomous_vehicle.agents import E2EAgent


def main():
    """Train E2E agent for highway driving."""
    print("=" * 60)
    print("Training E2E Agent for Highway Driving")
    print("=" * 60)
    
    # Create environment
    print("\n[1/4] Creating highway environment...")
    env = HighwayEnv()
    eval_env = HighwayEnv()
    
    # Create agent with PPO
    print("[2/4] Initializing PPO agent...")
    agent = E2EAgent(
        env=env,
        algorithm="PPO",
        policy="MlpPolicy",
        learning_rate=5e-4,
        total_timesteps=200000,
        log_dir="./logs/highway_e2e",
        model_dir="./models/highway_e2e",
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
    )
    
    # Train
    print("[3/4] Training agent...")
    agent.train(
        eval_env=eval_env,
        eval_freq=10000,
        checkpoint_freq=50000,
        n_eval_episodes=5,
    )
    
    # Save final model
    print("[4/4] Saving final model...")
    agent.save()
    
    # Evaluate
    print("\nEvaluating trained agent...")
    metrics = agent.evaluate(eval_env, n_episodes=10, render=False)
    
    print("\nEvaluation Results:")
    print(f"  Mean Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
    print(f"  Mean Episode Length: {metrics['mean_length']:.2f} ± {metrics['std_length']:.2f}")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    # Cleanup
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
