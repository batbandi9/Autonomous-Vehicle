"""
Tests for MPC controller.
"""

import pytest
import numpy as np
from autonomous_vehicle.controllers import MPCController


class TestMPCController:
    """Test MPC controller."""
    
    def test_initialization(self):
        """Test controller can be created."""
        controller = MPCController()
        assert controller is not None
        assert controller.horizon == 10
        assert controller.dt == 0.1
    
    def test_kinematic_model(self):
        """Test kinematic model."""
        controller = MPCController()
        state = np.array([0.0, 0.0, 0.0, 5.0])  # x, y, heading, speed
        control = np.array([0.0, 0.0])  # acceleration, steering
        
        next_state = controller.kinematic_model(state, control)
        assert len(next_state) == 4
        assert next_state[3] == 5.0  # Speed unchanged with zero acceleration
    
    def test_linearize_model(self):
        """Test model linearization."""
        controller = MPCController()
        state = np.array([0.0, 0.0, 0.0, 5.0])
        control = np.array([0.0, 0.0])
        
        A, B = controller.linearize_model(state, control)
        assert A.shape == (4, 4)
        assert B.shape == (4, 2)
    
    def test_compute_reference_trajectory(self):
        """Test reference trajectory generation."""
        controller = MPCController(horizon=5)
        current_state = np.array([0.0, 0.0, 0.0, 0.0])
        goal_state = np.array([10.0, 10.0, 0.0, 5.0])
        
        trajectory = controller.compute_reference_trajectory(current_state, goal_state)
        assert len(trajectory) == 5
        assert all(len(state) == 4 for state in trajectory)
    
    def test_solve(self):
        """Test MPC solve."""
        controller = MPCController(horizon=5)
        current_state = np.array([0.0, 0.0, 0.0, 5.0])
        goal_state = np.array([10.0, 0.0, 0.0, 5.0])
        
        reference = controller.compute_reference_trajectory(current_state, goal_state)
        control, trajectory = controller.solve(current_state, reference)
        
        assert len(control) == 2
        assert len(trajectory) > 0
