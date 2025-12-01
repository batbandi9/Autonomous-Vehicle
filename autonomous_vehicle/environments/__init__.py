"""
Environment wrappers and custom environments for autonomous vehicle training.
"""

__all__ = ["ParkingEnv", "LaneChangeEnv", "HighwayEnv"]


def __getattr__(name):
    """Lazy import to avoid loading dependencies until needed."""
    if name == "ParkingEnv":
        from autonomous_vehicle.environments.parking_env import ParkingEnv
        return ParkingEnv
    elif name == "LaneChangeEnv":
        from autonomous_vehicle.environments.lane_change_env import LaneChangeEnv
        return LaneChangeEnv
    elif name == "HighwayEnv":
        from autonomous_vehicle.environments.highway_env import HighwayEnv
        return HighwayEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
