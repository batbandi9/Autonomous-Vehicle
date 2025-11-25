"""
Example: Training HRL agent with MPC for parking.

This script demonstrates hierarchical RL combined with MPC
for the parking task.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from autonomous_vehicle.environments import ParkingEnv
from autonomous_vehicle.agents import HRLAgent


def main():
    """Train HRL agent with MPC for parking."""
    print("=" * 60)
    print("Training HRL+MPC Agent for Parking")
    print("=" * 60)
    
    # Create environment
    print("\n[1/4] Creating parking environment...")
    env = ParkingEnv()
    eval_env = ParkingEnv()
    
    # Create HRL agent
    print("[2/4] Initializing HRL agent with MPC...")
    agent = HRLAgent(
        env=env,
        algorithm="PPO",
        mpc_horizon=10,
        mpc_dt=0.1,
        decision_frequency=5,
        learning_rate=3e-4,
        total_timesteps=100000,
        log_dir="./logs/parking_hrl",
        model_dir="./models/parking_hrl",
    )
    
    # Train
    print("[3/4] Training HRL agent (high-level policy)...")
    agent.train(
        eval_env=eval_env,
        eval_freq=10000,
        checkpoint_freq=25000,
        n_eval_episodes=5,
    )
    
    # Save final model
    print("[4/4] Saving final model...")
    agent.save()
    
    # Evaluate with MPC
    print("\nEvaluating HRL+MPC agent...")
    metrics = agent.evaluate(eval_env, n_episodes=10, use_mpc=True)
    
    print("\nEvaluation Results (with MPC):")
    print(f"  Mean Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
    print(f"  Mean Episode Length: {metrics['mean_length']:.2f} ± {metrics['std_length']:.2f}")
    
    # Compare with high-level only
    print("\nEvaluating high-level policy only (without MPC)...")
    metrics_hl = agent.evaluate(eval_env, n_episodes=10, use_mpc=False)
    
    print("\nEvaluation Results (without MPC):")
    print(f"  Mean Reward: {metrics_hl['mean_reward']:.2f} ± {metrics_hl['std_reward']:.2f}")
    print(f"  Mean Episode Length: {metrics_hl['mean_length']:.2f} ± {metrics_hl['std_length']:.2f}")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    # Cleanup
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
