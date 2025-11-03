"""
Reinforcement Learning agents for autonomous vehicle control.
"""

__all__ = ["E2EAgent", "HRLAgent"]


def __getattr__(name):
    """Lazy import to avoid loading dependencies until needed."""
    if name == "E2EAgent":
        from autonomous_vehicle.agents.e2e_agent import E2EAgent
        return E2EAgent
    elif name == "HRLAgent":
        from autonomous_vehicle.agents.hrl_agent import HRLAgent
        return HRLAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
