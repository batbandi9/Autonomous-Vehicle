"""
End-to-End (E2E) Reinforcement Learning Agent.

This module provides a wrapper for training various RL algorithms
(DQN, PPO, SAC, etc.) using Stable Baselines3.
"""

import os
from typing import Dict, Any, Optional, Type
import numpy as np
from stable_baselines3 import PPO, DQN, SAC, A2C, TD3
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor


class E2EAgent:
    """
    End-to-End RL agent for autonomous driving.
    
    Supports multiple algorithms: PPO, DQN, SAC, A2C, TD3
    """
    
    ALGORITHMS = {
        "PPO": PPO,
        "DQN": DQN,
        "SAC": SAC,
        "A2C": A2C,
        "TD3": TD3,
    }
    
    def __init__(
        self,
        env,
        algorithm: str = "PPO",
        policy: str = "MlpPolicy",
        learning_rate: float = 3e-4,
        total_timesteps: int = 100000,
        log_dir: str = "./logs",
        model_dir: str = "./models",
        **kwargs
    ):
        """
        Initialize E2E agent.
        
        Args:
            env: Gymnasium environment
            algorithm: RL algorithm to use (PPO, DQN, SAC, A2C, TD3)
            policy: Policy network type
            learning_rate: Learning rate for optimizer
            total_timesteps: Total training timesteps
            log_dir: Directory for tensorboard logs
            model_dir: Directory to save models
            **kwargs: Additional algorithm-specific parameters
        """
        self.algorithm_name = algorithm
        self.policy = policy
        self.learning_rate = learning_rate
        self.total_timesteps = total_timesteps
        self.log_dir = log_dir
        self.model_dir = model_dir
        
        # Create directories
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        
        # Wrap environment
        self.env = Monitor(env)
        self.vec_env = DummyVecEnv([lambda: self.env])
        
        # Get algorithm class
        if algorithm not in self.ALGORITHMS:
            raise ValueError(
                f"Algorithm {algorithm} not supported. "
                f"Choose from {list(self.ALGORITHMS.keys())}"
            )
        
        AlgorithmClass = self.ALGORITHMS[algorithm]
        
        # Create model
        self.model = AlgorithmClass(
            policy,
            self.vec_env,
            learning_rate=learning_rate,
            tensorboard_log=log_dir,
            verbose=1,
            **kwargs
        )
        
    def train(
        self,
        eval_env=None,
        eval_freq: int = 10000,
        checkpoint_freq: int = 50000,
        n_eval_episodes: int = 5,
    ):
        """
        Train the agent.
        
        Args:
            eval_env: Environment for evaluation during training
            eval_freq: Frequency of evaluation in timesteps
            checkpoint_freq: Frequency of saving checkpoints
            n_eval_episodes: Number of episodes for each evaluation
        """
        callbacks = []
        
        # Evaluation callback
        if eval_env is not None:
            eval_env = Monitor(eval_env)
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=os.path.join(self.model_dir, "best_model"),
                log_path=self.log_dir,
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                deterministic=True,
                render=False,
            )
            callbacks.append(eval_callback)
        
        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=os.path.join(self.model_dir, "checkpoints"),
            name_prefix=f"{self.algorithm_name}_model",
        )
        callbacks.append(checkpoint_callback)
        
        # Train
        callback_list = CallbackList(callbacks) if callbacks else None
        self.model.learn(
            total_timesteps=self.total_timesteps,
            callback=callback_list,
            progress_bar=True,
        )
        
    def save(self, path: Optional[str] = None):
        """
        Save the trained model.
        
        Args:
            path: Path to save the model. If None, uses default model_dir
        """
        if path is None:
            path = os.path.join(self.model_dir, f"{self.algorithm_name}_final")
        self.model.save(path)
        print(f"Model saved to {path}")
        
    def load(self, path: str):
        """
        Load a trained model.
        
        Args:
            path: Path to the saved model
        """
        AlgorithmClass = self.ALGORITHMS[self.algorithm_name]
        self.model = AlgorithmClass.load(path, env=self.vec_env)
        print(f"Model loaded from {path}")
        
    def predict(self, observation: np.ndarray, deterministic: bool = True):
        """
        Predict action for given observation.
        
        Args:
            observation: Current observation
            deterministic: Whether to use deterministic policy
            
        Returns:
            Predicted action and state
        """
        return self.model.predict(observation, deterministic=deterministic)
    
    def evaluate(
        self,
        env,
        n_episodes: int = 10,
        deterministic: bool = True,
        render: bool = False,
    ) -> Dict[str, float]:
        """
        Evaluate the agent.
        
        Args:
            env: Environment for evaluation
            n_episodes: Number of episodes to evaluate
            deterministic: Whether to use deterministic policy
            render: Whether to render episodes
            
        Returns:
            Dictionary with evaluation metrics
        """
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(n_episodes):
            obs, info = env.reset()
            done = False
            truncated = False
            episode_reward = 0
            episode_length = 0
            
            while not (done or truncated):
                action, _ = self.predict(obs, deterministic=deterministic)
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                episode_length += 1
                
                if render:
                    env.render()
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        return {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "mean_length": np.mean(episode_lengths),
            "std_length": np.std(episode_lengths),
        }
