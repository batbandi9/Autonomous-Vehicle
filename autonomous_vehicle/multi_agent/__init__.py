"""
Multi-agent environments for coordinated autonomous driving.
"""

__all__ = ["MultiParkingEnv", "MultiHighwayEnv"]


def __getattr__(name):
    """Lazy import to avoid loading dependencies until needed."""
    if name == "MultiParkingEnv":
        from autonomous_vehicle.multi_agent.multi_parking_env import MultiParkingEnv
        return MultiParkingEnv
    elif name == "MultiHighwayEnv":
        from autonomous_vehicle.multi_agent.multi_highway_env import MultiHighwayEnv
        return MultiHighwayEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
