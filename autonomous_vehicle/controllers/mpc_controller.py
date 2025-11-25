"""
Model Predictive Control (MPC) for low-level vehicle control.

This module implements MPC for trajectory tracking and control stability.
"""

import numpy as np
import cvxpy as cp
from typing import Dict, Any, Optional, List, Tuple


class MPCController:
    """
    Model Predictive Controller for autonomous vehicle control.
    
    Uses a kinematic bicycle model for vehicle dynamics and solves
    an optimization problem to compute optimal control inputs.
    """
    
    def __init__(
        self,
        horizon: int = 10,
        dt: float = 0.1,
        wheelbase: float = 2.5,
        max_speed: float = 30.0,
        max_accel: float = 3.0,
        max_steer: float = np.deg2rad(30),
        Q: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
    ):
        """
        Initialize MPC controller.
        
        Args:
            horizon: Prediction horizon (number of steps)
            dt: Time step (seconds)
            wheelbase: Vehicle wheelbase length (meters)
            max_speed: Maximum speed (m/s)
            max_accel: Maximum acceleration (m/s^2)
            max_steer: Maximum steering angle (radians)
            Q: State cost matrix (4x4)
            R: Control cost matrix (2x2)
        """
        self.horizon = horizon
        self.dt = dt
        self.wheelbase = wheelbase
        self.max_speed = max_speed
        self.max_accel = max_accel
        self.max_steer = max_steer
        
        # State: [x, y, heading, speed]
        # Control: [acceleration, steering]
        self.state_dim = 4
        self.control_dim = 2
        
        # Cost matrices
        if Q is None:
            self.Q = np.diag([1.0, 1.0, 0.5, 0.1])  # State cost
        else:
            self.Q = Q
            
        if R is None:
            self.R = np.diag([0.1, 0.1])  # Control cost
        else:
            self.R = R
        
    def kinematic_model(
        self,
        state: np.ndarray,
        control: np.ndarray
    ) -> np.ndarray:
        """
        Kinematic bicycle model for vehicle dynamics.
        
        Args:
            state: Current state [x, y, heading, speed]
            control: Control input [acceleration, steering]
            
        Returns:
            Next state
        """
        x, y, theta, v = state
        a, delta = control
        
        # Update state using kinematic bicycle model
        x_next = x + v * np.cos(theta) * self.dt
        y_next = y + v * np.sin(theta) * self.dt
        theta_next = theta + (v / self.wheelbase) * np.tan(delta) * self.dt
        v_next = v + a * self.dt
        
        # Normalize heading to [-pi, pi]
        theta_next = np.arctan2(np.sin(theta_next), np.cos(theta_next))
        
        return np.array([x_next, y_next, theta_next, v_next])
    
    def linearize_model(
        self,
        state: np.ndarray,
        control: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Linearize the kinematic model around a state and control.
        
        Args:
            state: State to linearize around
            control: Control to linearize around
            
        Returns:
            A matrix (state transition) and B matrix (control input)
        """
        x, y, theta, v = state
        a, delta = control
        
        # State transition matrix A
        A = np.array([
            [1, 0, -v * np.sin(theta) * self.dt, np.cos(theta) * self.dt],
            [0, 1, v * np.cos(theta) * self.dt, np.sin(theta) * self.dt],
            [0, 0, 1, np.tan(delta) / self.wheelbase * self.dt],
            [0, 0, 0, 1]
        ])
        
        # Control input matrix B
        B = np.array([
            [0, 0],
            [0, 0],
            [0, v / (self.wheelbase * np.cos(delta)**2) * self.dt],
            [self.dt, 0]
        ])
        
        return A, B
    
    def solve(
        self,
        current_state: np.ndarray,
        reference_trajectory: List[np.ndarray],
        warm_start_controls: Optional[List[np.ndarray]] = None
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Solve MPC optimization problem.
        
        Args:
            current_state: Current vehicle state [x, y, heading, speed]
            reference_trajectory: List of reference states for the horizon
            warm_start_controls: Optional warm start for control sequence
            
        Returns:
            Optimal control input and predicted trajectory
        """
        # Ensure reference trajectory has correct length
        if len(reference_trajectory) < self.horizon:
            # Extend with last state
            last_state = reference_trajectory[-1]
            reference_trajectory.extend(
                [last_state] * (self.horizon - len(reference_trajectory))
            )
        
        # Define optimization variables
        states = [cp.Variable(self.state_dim) for _ in range(self.horizon + 1)]
        controls = [cp.Variable(self.control_dim) for _ in range(self.horizon)]
        
        # Initial condition constraint
        constraints = [states[0] == current_state]
        
        # Cost function
        cost = 0
        
        for t in range(self.horizon):
            # State cost
            state_error = states[t] - reference_trajectory[t]
            cost += cp.quad_form(state_error, self.Q)
            
            # Control cost
            cost += cp.quad_form(controls[t], self.R)
            
            # Dynamics constraints (linearized around reference)
            A, B = self.linearize_model(
                reference_trajectory[t],
                warm_start_controls[t] if warm_start_controls else np.zeros(2)
            )
            
            # Linearized dynamics: x_{t+1} = A*x_t + B*u_t
            constraints.append(
                states[t + 1] == A @ states[t] + B @ controls[t]
            )
            
            # Control constraints
            constraints.append(controls[t][0] >= -self.max_accel)  # Min accel
            constraints.append(controls[t][0] <= self.max_accel)   # Max accel
            constraints.append(controls[t][1] >= -self.max_steer)  # Min steer
            constraints.append(controls[t][1] <= self.max_steer)   # Max steer
            
            # Speed constraints
            constraints.append(states[t + 1][3] >= 0)  # Min speed
            constraints.append(states[t + 1][3] <= self.max_speed)  # Max speed
        
        # Terminal cost
        terminal_error = states[self.horizon] - reference_trajectory[self.horizon - 1]
        cost += cp.quad_form(terminal_error, self.Q)
        
        # Solve optimization problem
        problem = cp.Problem(cp.Minimize(cost), constraints)
        
        try:
            problem.solve(solver=cp.OSQP, warm_start=warm_start_controls is not None)
            
            if problem.status not in ["optimal", "optimal_inaccurate"]:
                print(f"Warning: MPC solver status: {problem.status}")
                # Return zero control if solver fails
                return np.zeros(self.control_dim), [current_state]
            
            # Extract optimal control and trajectory
            optimal_control = controls[0].value
            predicted_trajectory = [s.value for s in states]
            
            return optimal_control, predicted_trajectory
            
        except Exception as e:
            print(f"MPC solver error: {e}")
            return np.zeros(self.control_dim), [current_state]
    
    def compute_reference_trajectory(
        self,
        current_state: np.ndarray,
        goal_state: np.ndarray,
    ) -> List[np.ndarray]:
        """
        Generate a simple reference trajectory from current to goal state.
        
        Args:
            current_state: Current vehicle state
            goal_state: Goal vehicle state
            
        Returns:
            List of reference states
        """
        trajectory = []
        for i in range(self.horizon):
            alpha = (i + 1) / self.horizon
            # Linear interpolation
            ref_state = (1 - alpha) * current_state + alpha * goal_state
            trajectory.append(ref_state)
        
        return trajectory
