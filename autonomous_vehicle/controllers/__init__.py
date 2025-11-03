"""
Controllers for autonomous vehicle low-level control.
"""

__all__ = ["MPCController"]


def __getattr__(name):
    """Lazy import to avoid loading dependencies until needed."""
    if name == "MPCController":
        from autonomous_vehicle.controllers.mpc_controller import MPCController
        return MPCController
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
