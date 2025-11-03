"""
Autonomous Vehicle Learning Framework

This package provides tools for training autonomous driving agents using:
- End-to-End (E2E) Reinforcement Learning
- Hierarchical Reinforcement Learning (HRL)
- Model Predictive Control (MPC) integration
- Multi-agent scenarios
"""

__version__ = "0.1.0"

from autonomous_vehicle.environments import (
    ParkingEnv,
    LaneChangeEnv,
    HighwayEnv,
)

__all__ = [
    "ParkingEnv",
    "LaneChangeEnv",
    "HighwayEnv",
]
