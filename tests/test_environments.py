"""
Basic tests for autonomous vehicle environments.
"""

import pytest
import numpy as np
from autonomous_vehicle.environments import ParkingEnv, LaneChangeEnv, HighwayEnv


class TestParkingEnv:
    """Test parking environment."""
    
    def test_initialization(self):
        """Test environment can be created."""
        env = ParkingEnv()
        assert env is not None
        env.close()
    
    def test_reset(self):
        """Test environment can be reset."""
        env = ParkingEnv()
        obs, info = env.reset()
        assert obs is not None
        assert isinstance(info, dict)
        env.close()
    
    def test_step(self):
        """Test environment step."""
        env = ParkingEnv()
        obs, info = env.reset()
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        assert obs is not None
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(truncated, bool)
        env.close()
    
    def test_get_state(self):
        """Test getting vehicle state."""
        env = ParkingEnv()
        env.reset()
        state = env.get_state()
        assert "position" in state
        assert "velocity" in state
        assert "heading" in state
        assert "speed" in state
        env.close()


class TestLaneChangeEnv:
    """Test lane change environment."""
    
    def test_initialization(self):
        """Test environment can be created."""
        env = LaneChangeEnv()
        assert env is not None
        env.close()
    
    def test_reset(self):
        """Test environment can be reset."""
        env = LaneChangeEnv()
        obs, info = env.reset()
        assert obs is not None
        env.close()
    
    def test_step(self):
        """Test environment step."""
        env = LaneChangeEnv()
        env.reset()
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        assert obs is not None
        env.close()


class TestHighwayEnv:
    """Test highway environment."""
    
    def test_initialization(self):
        """Test environment can be created."""
        env = HighwayEnv()
        assert env is not None
        env.close()
    
    def test_reset(self):
        """Test environment can be reset."""
        env = HighwayEnv()
        obs, info = env.reset()
        assert obs is not None
        env.close()
    
    def test_step(self):
        """Test environment step."""
        env = HighwayEnv()
        env.reset()
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        assert obs is not None
        env.close()
    
    def test_get_road_state(self):
        """Test getting road state."""
        env = HighwayEnv()
        env.reset()
        road_state = env.get_road_state()
        assert "lanes_count" in road_state
        assert "vehicles" in road_state
        env.close()
