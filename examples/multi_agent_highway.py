"""
Example: Multi-agent highway driving.

This script demonstrates multi-agent highway driving scenario.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from autonomous_vehicle.multi_agent import MultiHighwayEnv
import numpy as np


def main():
    """Multi-agent highway driving example."""
    print("=" * 60)
    print("Multi-Agent Highway Driving Example")
    print("=" * 60)
    
    # Create multi-agent environment
    print("\n[1/3] Creating multi-agent highway environment...")
    n_agents = 3
    env = MultiHighwayEnv(n_agents=n_agents)
    
    print(f"Number of controlled agents: {n_agents}")
    print(f"Agents: {env.agents}")
    
    # Reset environment
    print("\n[2/3] Resetting environment...")
    observations, info = env.reset()
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Run a few steps with random actions
    print("\n[3/3] Running test episode with random actions...")
    total_reward = {agent: 0 for agent in env.agents}
    
    for step in range(50):
        # Random actions for all agents
        actions = {
            agent: env.action_space.sample()
            for agent in env.agents
        }
        
        observations, rewards, dones, truncateds, info = env.step(actions)
        
        for agent in env.agents:
            total_reward[agent] += rewards[agent]
        
        if step % 10 == 0:
            # Get agent states
            states = env.get_agent_states()
            print(f"\nStep {step}:")
            for agent, state in states.items():
                print(f"  {agent}: speed={state['speed']:.2f} m/s, "
                      f"lane={state.get('lane_index', 'N/A')}")
        
        if dones["__all__"] or truncateds["__all__"]:
            print(f"\nEpisode ended at step {step + 1}")
            break
    
    print("\nTotal rewards per agent:")
    for agent, reward in total_reward.items():
        print(f"  {agent}: {reward:.2f}")
    
    print("\n" + "=" * 60)
    print("Multi-Agent Highway Example Complete!")
    print("=" * 60)
    print("\nNote: For actual training, integrate with multi-agent RL frameworks")
    print("such as RLlib or PettingZoo's parallel API.")
    
    # Cleanup
    env.close()


if __name__ == "__main__":
    main()
