"""
Multi-agent highway environment.

Supports multiple vehicles learning cooperative highway driving.
"""

import gymnasium as gym
import numpy as np
from typing import Dict, Any, List, Tuple, Optional


class MultiHighwayEnv:
    """
    Multi-agent highway environment wrapper.
    
    Manages multiple vehicles on a highway that must coordinate for
    safe and efficient driving.
    """
    
    def __init__(
        self,
        n_agents: int = 3,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize multi-agent highway environment.
        
        Args:
            n_agents: Number of controlled agents/vehicles
            config: Environment configuration
        """
        self.n_agents = n_agents
        self.agents = [f"agent_{i}" for i in range(n_agents)]
        
        default_config = {
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 5,
                "features": ["presence", "x", "y", "vx", "vy"],
                "absolute": False,
                "normalize": True,
                "order": "sorted"
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "lanes_count": 4,
            "vehicles_count": 50,
            "controlled_vehicles": n_agents,
            "duration": 40,
            "collision_reward": -1,
            "right_lane_reward": 0.1,
            "high_speed_reward": 0.4,
            "lane_change_reward": 0,
            "reward_speed_range": [20, 30],
            "normalize_reward": True,
            "simulation_frequency": 15,
            "policy_frequency": 1,
        }
        
        if config:
            default_config.update(config)
        
        self.env = gym.make("highway-v0", render_mode=None)
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
        if isinstance(obs, np.ndarray) and len(obs.shape) > 1:
            # Multi-agent observation case
            for i, agent in enumerate(self.agents):
                if i < len(obs):
                    observations[agent] = obs[i]
                else:
                    observations[agent] = np.zeros_like(obs[0])
        else:
            # Single observation case - replicate
            for agent in self.agents:
                observations[agent] = obs
        
        return observations, info
    
    def step(
        self,
        actions: Dict[str, int]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict]:
        """
        Execute actions for all agents.
        
        Args:
            actions: Dictionary of actions for each agent
            
        Returns:
            Observations, rewards, dones, truncated, and info for each agent
        """
        # Combine actions from all agents
        combined_action = [actions[agent] for agent in self.agents]
        
        obs, reward, done, truncated, info = self.env.step(combined_action)
        
        # Split observations
        observations = {}
        if isinstance(obs, np.ndarray) and len(obs.shape) > 1:
            for i, agent in enumerate(self.agents):
                if i < len(obs):
                    observations[agent] = obs[i]
                else:
                    observations[agent] = np.zeros_like(obs[0])
        else:
            for agent in self.agents:
                observations[agent] = obs
        
        # Create per-agent rewards
        if isinstance(reward, (list, np.ndarray)) and len(reward) == self.n_agents:
            rewards = {agent: reward[i] for i, agent in enumerate(self.agents)}
        else:
            # Shared reward
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
        return self.env.observation_space
    
    @property
    def action_space(self):
        """Get action space for a single agent."""
        return self.env.action_space
    
    def get_agent_states(self) -> Dict[str, Dict[str, Any]]:
        """
        Get states of all controlled agents.
        
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
                    "lane_index": vehicle.lane_index,
                }
        
        return states
    
    def get_traffic_info(self) -> List[Dict[str, Any]]:
        """
        Get information about all vehicles in the environment.
        
        Returns:
            List of vehicle information dictionaries
        """
        traffic = []
        for vehicle in self.env.unwrapped.road.vehicles:
            traffic.append({
                "position": vehicle.position,
                "velocity": vehicle.velocity,
                "heading": vehicle.heading,
                "speed": vehicle.speed,
            })
        
        return traffic
