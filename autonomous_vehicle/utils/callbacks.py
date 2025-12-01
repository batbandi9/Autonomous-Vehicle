"""
Custom callbacks for training monitoring.
"""

from stable_baselines3.common.callbacks import BaseCallback
import numpy as np


class CustomCallback(BaseCallback):
    """
    Custom callback for monitoring training progress.
    """
    
    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        """
        Called at every step.
        
        Returns:
            If the callback returns False, training is aborted early.
        """
        return True
    
    def _on_rollout_end(self) -> None:
        """
        Called at the end of a rollout.
        """
        if self.verbose > 0:
            print(f"Rollout ended at step {self.num_timesteps}")
