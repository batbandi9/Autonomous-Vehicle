"""
Hierarchical Reinforcement Learning (HRL) Agent.

This module implements HRL by combining high-level RL decision making
with low-level MPC control for improved safety and sample efficiency.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from autonomous_vehicle.agents.e2e_agent import E2EAgent
from autonomous_vehicle.controllers.mpc_controller import MPCController


class HRLAgent:
    """
    Hierarchical RL agent combining high-level RL with low-level MPC.
    
    The high-level RL agent makes strategic decisions (e.g., which lane to use,
    when to change lanes, target speed), while the low-level MPC controller
    ensures safe and smooth execution of these decisions.
    """
    
    def __init__(
        self,
        env,
        algorithm: str = "PPO",
        mpc_horizon: int = 10,
        mpc_dt: float = 0.1,
        decision_frequency: int = 5,
        log_dir: str = "./logs/hrl",
        model_dir: str = "./models/hrl",
        **kwargs
    ):
        """
        Initialize HRL agent.
        
        Args:
            env: Gymnasium environment
            algorithm: RL algorithm for high-level policy
            mpc_horizon: MPC prediction horizon
            mpc_dt: MPC time step
            decision_frequency: How often (in env steps) to make high-level decisions
            log_dir: Directory for logs
            model_dir: Directory for models
            **kwargs: Additional parameters for E2E agent
        """
        # High-level RL agent
        self.high_level_agent = E2EAgent(
            env=env,
            algorithm=algorithm,
            log_dir=log_dir,
            model_dir=model_dir,
            **kwargs
        )
        
        # Low-level MPC controller
        self.mpc_controller = MPCController(
            horizon=mpc_horizon,
            dt=mpc_dt,
        )
        
        self.decision_frequency = decision_frequency
        self.step_counter = 0
        self.current_high_level_action = None
        
    def _high_level_to_goal(
        self,
        high_level_action: np.ndarray,
        current_state: Dict[str, Any]
    ) -> np.ndarray:
        """
        Convert high-level action to goal state for MPC.
        
        Args:
            high_level_action: Action from high-level RL policy
            current_state: Current vehicle state
            
        Returns:
            Goal state for MPC [x, y, heading, speed]
        """
        # Extract current position and heading
        pos = current_state.get("position", np.array([0.0, 0.0]))
        heading = current_state.get("heading", 0.0)
        speed = current_state.get("speed", 0.0)
        
        # Interpret high-level action as target speed and heading change
        # This is a simple interpretation - can be customized based on action space
        if isinstance(high_level_action, np.ndarray) and len(high_level_action) >= 2:
            target_speed = np.clip(high_level_action[0], 0, 30)
            heading_change = high_level_action[1] * 0.5  # Scale heading change
        else:
            # Discrete action case
            target_speed = speed
            heading_change = 0
        
        # Compute goal position (simple forward projection)
        horizon_distance = target_speed * self.mpc_controller.horizon * self.mpc_controller.dt
        goal_x = pos[0] + horizon_distance * np.cos(heading + heading_change)
        goal_y = pos[1] + horizon_distance * np.sin(heading + heading_change)
        goal_heading = heading + heading_change
        
        return np.array([goal_x, goal_y, goal_heading, target_speed])
    
    def predict(
        self,
        observation: np.ndarray,
        env_state: Optional[Dict[str, Any]] = None,
        deterministic: bool = True
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Predict action using hierarchical control.
        
        Args:
            observation: Current observation
            env_state: Current environment state (for MPC)
            deterministic: Whether to use deterministic policy
            
        Returns:
            Low-level control action and additional info
        """
        # Update high-level action if needed
        if self.step_counter % self.decision_frequency == 0:
            self.current_high_level_action, _ = self.high_level_agent.predict(
                observation, deterministic=deterministic
            )
        
        self.step_counter += 1
        
        # If no environment state provided, return high-level action directly
        if env_state is None:
            return self.current_high_level_action, {}
        
        # Convert environment state to MPC state format
        pos = env_state.get("position", np.array([0.0, 0.0]))
        heading = env_state.get("heading", 0.0)
        speed = env_state.get("speed", 0.0)
        current_mpc_state = np.array([pos[0], pos[1], heading, speed])
        
        # Get goal state from high-level action
        goal_state = self._high_level_to_goal(
            self.current_high_level_action,
            env_state
        )
        
        # Generate reference trajectory
        reference_trajectory = self.mpc_controller.compute_reference_trajectory(
            current_mpc_state,
            goal_state
        )
        
        # Solve MPC
        mpc_control, predicted_trajectory = self.mpc_controller.solve(
            current_mpc_state,
            reference_trajectory
        )
        
        # Convert MPC control to environment action
        # MPC outputs [acceleration, steering], may need conversion
        # depending on environment action space
        action = mpc_control
        
        info = {
            "high_level_action": self.current_high_level_action,
            "mpc_control": mpc_control,
            "predicted_trajectory": predicted_trajectory,
        }
        
        return action, info
    
    def train(self, **kwargs):
        """
        Train the high-level RL agent.
        
        Args:
            **kwargs: Training parameters for E2E agent
        """
        self.high_level_agent.train(**kwargs)
    
    def save(self, path: Optional[str] = None):
        """Save the high-level agent."""
        self.high_level_agent.save(path)
    
    def load(self, path: str):
        """Load the high-level agent."""
        self.high_level_agent.load(path)
    
    def evaluate(
        self,
        env,
        n_episodes: int = 10,
        deterministic: bool = True,
        use_mpc: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate the HRL agent.
        
        Args:
            env: Environment for evaluation
            n_episodes: Number of episodes
            deterministic: Whether to use deterministic policy
            use_mpc: Whether to use MPC for low-level control
            
        Returns:
            Evaluation metrics
        """
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(n_episodes):
            obs, info = env.reset()
            done = False
            truncated = False
            episode_reward = 0
            episode_length = 0
            self.step_counter = 0
            
            while not (done or truncated):
                if use_mpc and hasattr(env, 'get_state'):
                    env_state = env.get_state()
                    action, _ = self.predict(obs, env_state, deterministic)
                else:
                    # Fallback to high-level only
                    action, _ = self.high_level_agent.predict(obs, deterministic)
                
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                episode_length += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        return {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "mean_length": np.mean(episode_lengths),
            "std_length": np.std(episode_lengths),
        }
