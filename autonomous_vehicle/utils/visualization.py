"""
Visualization utilities for training and evaluation.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any, Optional


def plot_training_results(
    log_file: str,
    save_path: Optional[str] = None
):
    """
    Plot training results from tensorboard logs.
    
    Args:
        log_file: Path to tensorboard log file
        save_path: Optional path to save the figure
    """
    # This is a placeholder - actual implementation would parse tensorboard logs
    print(f"Plotting training results from {log_file}")
    if save_path:
        print(f"Saving to {save_path}")


def plot_trajectory(
    trajectory: List[np.ndarray],
    reference: Optional[List[np.ndarray]] = None,
    title: str = "Vehicle Trajectory",
    save_path: Optional[str] = None
):
    """
    Plot vehicle trajectory.
    
    Args:
        trajectory: List of states [x, y, heading, speed]
        reference: Optional reference trajectory
        title: Plot title
        save_path: Optional path to save the figure
    """
    if not trajectory:
        print("Empty trajectory")
        return
    
    trajectory = np.array(trajectory)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Position plot (x, y)
    axes[0, 0].plot(trajectory[:, 0], trajectory[:, 1], 'b-', label='Actual', linewidth=2)
    if reference is not None:
        reference = np.array(reference)
        axes[0, 0].plot(reference[:, 0], reference[:, 1], 'r--', label='Reference', linewidth=2)
    axes[0, 0].set_xlabel('X Position (m)')
    axes[0, 0].set_ylabel('Y Position (m)')
    axes[0, 0].set_title('2D Trajectory')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    axes[0, 0].axis('equal')
    
    # Heading plot
    axes[0, 1].plot(trajectory[:, 2], 'b-', linewidth=2)
    if reference is not None:
        axes[0, 1].plot(reference[:, 2], 'r--', linewidth=2)
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Heading (rad)')
    axes[0, 1].set_title('Heading over Time')
    axes[0, 1].grid(True)
    
    # Speed plot
    axes[1, 0].plot(trajectory[:, 3], 'b-', linewidth=2)
    if reference is not None:
        axes[1, 0].plot(reference[:, 3], 'r--', linewidth=2)
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Speed (m/s)')
    axes[1, 0].set_title('Speed over Time')
    axes[1, 0].grid(True)
    
    # Distance to reference
    if reference is not None:
        distances = np.linalg.norm(trajectory[:, :2] - reference[:, :2], axis=1)
        axes[1, 1].plot(distances, 'g-', linewidth=2)
        axes[1, 1].set_xlabel('Time Step')
        axes[1, 1].set_ylabel('Distance (m)')
        axes[1, 1].set_title('Distance to Reference')
        axes[1, 1].grid(True)
    else:
        axes[1, 1].axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Trajectory plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_evaluation_metrics(
    metrics: Dict[str, Any],
    save_path: Optional[str] = None
):
    """
    Plot evaluation metrics.
    
    Args:
        metrics: Dictionary of metrics
        save_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    
    bars = ax.bar(metric_names, metric_values, color='skyblue', edgecolor='navy', linewidth=1.5)
    
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Evaluation Metrics', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=10)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metrics plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
