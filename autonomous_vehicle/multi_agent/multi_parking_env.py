"""
Multi-agent parking environment.

Supports multiple vehicles learning to park cooperatively in a shared space.
"""

import gymnasium as gym
import numpy as np
from typing import Dict, Any, List, Tuple, Optional


class MultiParkingEnv:
    """
    Multi-agent parking environment wrapper.
    
    Manages multiple parking environments where agents must coordinate
    to avoid collisions while parking.
    """
    
    def __init__(
        self,
        n_agents: int = 2,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize multi-agent parking environment.
        
        Args:
            n_agents: Number of agents/vehicles
            config: Environment configuration
        """
        self.n_agents = n_agents
        self.agents = [f"agent_{i}" for i in range(n_agents)]
        
        default_config = {
            "observation": {
                "type": "KinematicsGoal",
                "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
                "scales": [100, 100, 5, 5, 1, 1],
                "normalize": False
            },
            "action": {
                "type": "ContinuousAction"
            },
            "controlled_vehicles": n_agents,
            "vehicles_count": n_agents,
            "reward_weights": [1, 0.3, 0, 0, 0.02, 0.02],
            "success_goal_reward": 0.12,
            "collision_reward": -5,
            "steering_range": np.deg2rad(45),
            "simulation_frequency": 15,
            "policy_frequency": 5,
            "duration": 100,
        }
        
        if config:
            default_config.update(config)
        
        self.env = gym.make("parking-v0", render_mode=None)
        self.env.unwrapped.config.update(default_config)
        self.env.reset()
        
    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, np.ndarray], Dict]:
        """
        Reset environment for all agents.
        
        Returns:
            Dictionary of observations for each agent
        """
        if seed is not None:
            self.env.unwrapped.np_random, _ = gym.utils.seeding.np_random(seed)
        
        obs, info = self.env.reset()
        
        # Split observations for each agent
        observations = {}
        if isinstance(obs, np.ndarray):
            # Assuming stacked observations for all agents
            obs_per_agent = len(obs) // self.n_agents
            for i, agent in enumerate(self.agents):
                start_idx = i * obs_per_agent
                end_idx = start_idx + obs_per_agent
                observations[agent] = obs[start_idx:end_idx]
        else:
            # If obs is already dict-like or single obs, replicate
            for agent in self.agents:
                observations[agent] = obs
        
        return observations, info
    
    def step(
        self,
        actions: Dict[str, np.ndarray]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict]:
        """
        Execute actions for all agents.
        
        Args:
            actions: Dictionary of actions for each agent
            
        Returns:
            Observations, rewards, dones, truncated, and info for each agent
        """
        # Combine actions from all agents
        combined_action = np.concatenate([actions[agent] for agent in self.agents])
        
        obs, reward, done, truncated, info = self.env.step(combined_action)
        
        # Split observations
        observations = {}
        obs_per_agent = len(obs) // self.n_agents
        for i, agent in enumerate(self.agents):
            start_idx = i * obs_per_agent
            end_idx = start_idx + obs_per_agent
            observations[agent] = obs[start_idx:end_idx]
        
        # Create per-agent rewards
        # In cooperative parking, reward is shared
        rewards = {agent: reward for agent in self.agents}
        
        # Done flags
        dones = {agent: done for agent in self.agents}
        dones["__all__"] = done
        
        # Truncated flags
        truncateds = {agent: truncated for agent in self.agents}
        truncateds["__all__"] = truncated
        
        return observations, rewards, dones, truncateds, info
    
    def render(self):
        """Render the environment."""
        return self.env.render()
    
    def close(self):
        """Close the environment."""
        self.env.close()
    
    @property
    def observation_space(self):
        """Get observation space for a single agent."""
        full_space = self.env.observation_space
        # Divide by number of agents
        return gym.spaces.Box(
            low=full_space.low[:len(full_space.low)//self.n_agents],
            high=full_space.high[:len(full_space.high)//self.n_agents],
            dtype=full_space.dtype
        )
    
    @property
    def action_space(self):
        """Get action space for a single agent."""
        full_space = self.env.action_space
        # Divide by number of agents
        return gym.spaces.Box(
            low=full_space.low[:len(full_space.low)//self.n_agents],
            high=full_space.high[:len(full_space.high)//self.n_agents],
            dtype=full_space.dtype
        )
    
    def get_agent_states(self) -> Dict[str, Dict[str, Any]]:
        """
        Get states of all agents.
        
        Returns:
            Dictionary mapping agent names to their states
        """
        states = {}
        vehicles = self.env.unwrapped.controlled_vehicles
        
        for i, agent in enumerate(self.agents):
            if i < len(vehicles):
                vehicle = vehicles[i]
                states[agent] = {
                    "position": vehicle.position,
                    "velocity": vehicle.velocity,
                    "heading": vehicle.heading,
                    "speed": vehicle.speed,
                }
        
        return states
