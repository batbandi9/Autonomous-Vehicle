"""
Highway driving environment for autonomous vehicle training.

This environment simulates general highway driving with multiple vehicles
and various traffic scenarios.
"""

import gymnasium as gym
import numpy as np
from typing import Dict, Any, Tuple, Optional


class HighwayEnv:
    """
    Wrapper for highway driving environment with standardized interface.
    
    The environment simulates realistic highway driving scenarios with
    multiple vehicles, requiring the agent to navigate safely at high speeds.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize highway environment.
        
        Args:
            config: Optional configuration dictionary for environment customization
        """
        self.default_config = {
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
            "controlled_vehicles": 1,
            "initial_lane_id": None,
            "duration": 40,
            "ego_spacing": 2,
            "vehicles_density": 1,
            "collision_reward": -1,
            "right_lane_reward": 0.1,
            "high_speed_reward": 0.4,
            "lane_change_reward": 0,
            "reward_speed_range": [20, 30],
            "normalize_reward": True,
            "offroad_terminal": False,
            "simulation_frequency": 15,
            "policy_frequency": 1,
            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
            "screen_width": 600,
            "screen_height": 150,
            "centering_position": [0.3, 0.5],
            "scaling": 5.5,
            "show_trajectories": False,
            "render_agent": True,
            "offscreen_rendering": False
        }
        
        if config:
            self.default_config.update(config)
        
        self.env = gym.make("highway-v0", render_mode=None)
        self.env.unwrapped.config.update(self.default_config)
        self.env.reset()
        
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment."""
        if seed is not None:
            self.env.unwrapped.np_random, _ = gym.utils.seeding.np_random(seed)
        obs, info = self.env.reset()
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        return self.env.step(action)
    
    def render(self):
        """Render the environment."""
        return self.env.render()
    
    def close(self):
        """Close the environment."""
        self.env.close()
    
    @property
    def observation_space(self):
        """Get observation space."""
        return self.env.observation_space
    
    @property
    def action_space(self):
        """Get action space."""
        return self.env.action_space
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get current state of the vehicle for MPC integration.
        
        Returns:
            Dictionary containing vehicle state information
        """
        vehicle = self.env.unwrapped.vehicle
        return {
            "position": vehicle.position,
            "velocity": vehicle.velocity,
            "heading": vehicle.heading,
            "speed": vehicle.speed,
            "lane_index": vehicle.lane_index,
        }
    
    def get_road_state(self) -> Dict[str, Any]:
        """
        Get road and traffic state for planning.
        
        Returns:
            Dictionary containing road network and traffic information
        """
        return {
            "lanes_count": self.env.unwrapped.config["lanes_count"],
            "vehicles": [
                {
                    "position": v.position,
                    "velocity": v.velocity,
                    "heading": v.heading,
                }
                for v in self.env.unwrapped.road.vehicles
            ],
        }
    
    def set_config(self, config: Dict[str, Any]):
        """
        Update environment configuration.
        
        Args:
            config: Configuration dictionary to update
        """
        self.env.unwrapped.config.update(config)
