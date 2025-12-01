"""
Tests for multi-agent environments.
"""

import pytest
import numpy as np
from autonomous_vehicle.multi_agent import MultiParkingEnv, MultiHighwayEnv


class TestMultiParkingEnv:
    """Test multi-agent parking environment."""
    
    def test_initialization(self):
        """Test environment can be created."""
        env = MultiParkingEnv(n_agents=2)
        assert env is not None
        assert env.n_agents == 2
        assert len(env.agents) == 2
        env.close()
    
    def test_reset(self):
        """Test environment reset."""
        env = MultiParkingEnv(n_agents=2)
        observations, info = env.reset()
        assert isinstance(observations, dict)
        assert len(observations) == 2
        env.close()
    
    def test_step(self):
        """Test environment step."""
        env = MultiParkingEnv(n_agents=2)
        observations, info = env.reset()
        
        actions = {agent: env.action_space.sample() for agent in env.agents}
        observations, rewards, dones, truncateds, info = env.step(actions)
        
        assert len(observations) == 2
        assert len(rewards) == 2
        assert "__all__" in dones
        env.close()


class TestMultiHighwayEnv:
    """Test multi-agent highway environment."""
    
    def test_initialization(self):
        """Test environment can be created."""
        env = MultiHighwayEnv(n_agents=3)
        assert env is not None
        assert env.n_agents == 3
        env.close()
    
    def test_reset(self):
        """Test environment reset."""
        env = MultiHighwayEnv(n_agents=2)
        observations, info = env.reset()
        assert isinstance(observations, dict)
        env.close()
    
    def test_step(self):
        """Test environment step."""
        env = MultiHighwayEnv(n_agents=2)
        observations, info = env.reset()
        
        actions = {agent: env.action_space.sample() for agent in env.agents}
        observations, rewards, dones, truncateds, info = env.step(actions)
        
        assert len(observations) == 2
        assert len(rewards) == 2
        env.close()
