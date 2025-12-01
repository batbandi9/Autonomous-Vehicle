"""
Lane changing environment for autonomous vehicle training.

This environment focuses on highway lane changing maneuvers with
traffic interaction.
"""

import gymnasium as gym
import numpy as np
from typing import Dict, Any, Tuple, Optional


class LaneChangeEnv:
    """
    Wrapper for lane changing environment with standardized interface.
    
    The environment simulates highway driving where the agent must
    change lanes safely while maintaining speed.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize lane changing environment.
        
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
            "lanes_count": 3,
            "vehicles_count": 15,
            "duration": 40,
            "initial_spacing": 2,
            "collision_reward": -1,
            "right_lane_reward": 0.1,
            "high_speed_reward": 0.4,
            "lane_change_reward": 0,
            "reward_speed_range": [20, 30],
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
    
    def get_nearby_vehicles(self, distance: float = 50.0) -> list:
        """
        Get nearby vehicles for multi-agent coordination.
        
        Args:
            distance: Maximum distance to consider vehicles as nearby
            
        Returns:
            List of nearby vehicle information
        """
        ego_vehicle = self.env.unwrapped.vehicle
        nearby = []
        
        for vehicle in self.env.unwrapped.road.vehicles:
            if vehicle is ego_vehicle:
                continue
            dist = np.linalg.norm(vehicle.position - ego_vehicle.position)
            if dist <= distance:
                nearby.append({
                    "position": vehicle.position,
                    "velocity": vehicle.velocity,
                    "distance": dist,
                })
        
        return nearby
