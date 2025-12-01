"""
Parking environment for autonomous vehicle training.

This environment uses highway-env's parking environment with custom configurations
optimized for hierarchical learning and MPC integration.
"""

import gymnasium as gym
import numpy as np
from typing import Dict, Any, Tuple, Optional


class ParkingEnv:
    """
    Wrapper for parking environment with standardized interface.
    
    The environment simulates a parking lot where the agent must navigate
    and park in a designated spot while avoiding obstacles.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize parking environment.
        
        Args:
            config: Optional configuration dictionary for environment customization
        """
        self.default_config = {
            "observation": {
                "type": "KinematicsGoal",
                "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
                "scales": [100, 100, 5, 5, 1, 1],
                "normalize": False
            },
            "action": {
                "type": "ContinuousAction"
            },
            "reward_weights": [1, 0.3, 0, 0, 0.02, 0.02],
            "success_goal_reward": 0.12,
            "collision_reward": -5,
            "steering_range": np.deg2rad(45),
            "simulation_frequency": 15,
            "policy_frequency": 5,
            "duration": 100,
            "screen_width": 600,
            "screen_height": 300,
            "centering_position": [0.5, 0.5],
            "scaling": 7,
            "controlled_vehicles": 1,
        }
        
        if config:
            self.default_config.update(config)
        
        self.env = gym.make("parking-v0", render_mode=None)
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
        }
    
    def set_state(self, state: Dict[str, Any]):
        """
        Set vehicle state (useful for MPC trajectory optimization).
        
        Args:
            state: Dictionary containing vehicle state to set
        """
        vehicle = self.env.unwrapped.vehicle
        if "position" in state:
            vehicle.position = state["position"]
        if "velocity" in state:
            vehicle.velocity = state["velocity"]
        if "heading" in state:
            vehicle.heading = state["heading"]
