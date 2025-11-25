"""
Utility functions for autonomous vehicle training and evaluation.
"""

__all__ = ["plot_training_results", "plot_trajectory", "CustomCallback"]


def __getattr__(name):
    """Lazy import to avoid loading dependencies until needed."""
    if name == "plot_training_results":
        from autonomous_vehicle.utils.visualization import plot_training_results
        return plot_training_results
    elif name == "plot_trajectory":
        from autonomous_vehicle.utils.visualization import plot_trajectory
        return plot_trajectory
    elif name == "CustomCallback":
        from autonomous_vehicle.utils.callbacks import CustomCallback
        return CustomCallback
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
