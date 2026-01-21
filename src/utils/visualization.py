"""Visualization utilities for DTI prediction.

This module provides plotting and visualization functions for analyzing
model performance, attention weights, and predictions.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 4),
):
    """Plot training history.
    
    Args:
        history: Training history dictionary
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot loss
    epochs = range(1, len(history['train_loss']) + 1)
    axes[0].plot(epochs, history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(epochs, history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot metrics (if available)
    if 'val_metrics' in history and len(history['val_metrics']) > 0:
        # Extract metric values
        metric_names = [k for k in history['val_metrics'][0].keys() 
                       if isinstance(history['val_metrics'][0][k], (int, float))]
        
        if metric_names:
            # Plot first metric
            metric_name = metric_names[0]
            metric_values = [m[metric_name] for m in history['val_metrics']]
            
            axes[1].plot(epochs, metric_values, label=metric_name, marker='o', color='green')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel(metric_name.upper())
            axes[1].set_title(f'Validation {metric_name.upper()}')
            axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.show()


def plot_predictions(
    targets: np.ndarray,
    predictions: np.ndarray,
    title: str = 'Predictions vs Ground Truth',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 8),
):
    """Plot predictions against ground truth.
    
    Args:
        targets: Ground truth values
        predictions: Predicted values
        title: Plot title
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    # Scatter plot
    plt.scatter(targets, predictions, alpha=0.5, s=20)
    
    # Perfect prediction line
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    # Compute R²
    from sklearn.metrics import r2_score
    r2 = r2_score(targets, predictions)
    
    plt.xlabel('Ground Truth')
    plt.ylabel('Predictions')
    plt.title(f'{title}\nR² = {r2:.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Predictions plot saved to {save_path}")
    
    plt.show()


def plot_residuals(
    targets: np.ndarray,
    predictions: np.ndarray,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 4),
):
    """Plot residual analysis.
    
    Args:
        targets: Ground truth values
        predictions: Predicted values
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    residuals = predictions - targets
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Residual plot
    axes[0].scatter(predictions, residuals, alpha=0.5, s=20)
    axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0].set_xlabel('Predicted Values')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title('Residual Plot')
    axes[0].grid(True, alpha=0.3)
    
    # Histogram of residuals
    axes[1].hist(residuals.flatten(), bins=50, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Residuals')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Residuals')
    axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Residuals plot saved to {save_path}")
    
    plt.show()


def plot_attention_weights(
    attention_weights: torch.Tensor,
    title: str = 'Attention Weights',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
):
    """Plot attention weight heatmap.
    
    Args:
        attention_weights: Attention weights tensor
        title: Plot title
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()
    
    # If multi-head, average across heads
    if len(attention_weights.shape) > 2:
        attention_weights = attention_weights.mean(axis=0)
    
    plt.figure(figsize=figsize)
    
    sns.heatmap(
        attention_weights,
        cmap='viridis',
        cbar=True,
        square=True,
        annot=False,
        fmt='.2f',
    )
    
    plt.title(title)
    plt.xlabel('Key')
    plt.ylabel('Query')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Attention weights plot saved to {save_path}")
    
    plt.show()


def plot_uncertainty(
    predictions: np.ndarray,
    uncertainties: np.ndarray,
    targets: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 4),
):
    """Plot uncertainty estimates.
    
    Args:
        predictions: Predicted values
        uncertainties: Uncertainty estimates
        targets: Ground truth values (optional)
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot predictions with error bars
    indices = np.arange(len(predictions))
    axes[0].errorbar(
        indices,
        predictions.flatten(),
        yerr=uncertainties.flatten(),
        fmt='o',
        alpha=0.6,
        capsize=3,
        label='Predictions ± Uncertainty',
    )
    
    if targets is not None:
        axes[0].scatter(indices, targets.flatten(), color='red', marker='x', s=50, label='Ground Truth')
        axes[0].legend()
    
    axes[0].set_xlabel('Sample Index')
    axes[0].set_ylabel('Value')
    axes[0].set_title('Predictions with Uncertainty')
    axes[0].grid(True, alpha=0.3)
    
    # Plot uncertainty distribution
    axes[1].hist(uncertainties.flatten(), bins=50, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Uncertainty')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Uncertainties')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Uncertainty plot saved to {save_path}")
    
    plt.show()


def plot_cold_start_comparison(
    results: Dict[str, Dict[str, float]],
    metric: str = 'rmse',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
):
    """Plot comparison of cold-start scenario performance.
    
    Args:
        results: Dictionary mapping scenario to metrics
        metric: Metric to plot
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    scenarios = list(results.keys())
    values = [results[s].get(metric, 0.0) for s in scenarios]
    
    plt.figure(figsize=figsize)
    
    bars = plt.bar(scenarios, values, color='steelblue', alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{height:.4f}',
            ha='center',
            va='bottom',
        )
    
    plt.xlabel('Scenario')
    plt.ylabel(metric.upper())
    plt.title(f'Cold-Start Performance Comparison ({metric.upper()})')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Cold-start comparison plot saved to {save_path}")
    
    plt.show()


def plot_learning_curves(
    train_sizes: List[int],
    train_scores: List[float],
    val_scores: List[float],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
):
    """Plot learning curves.
    
    Args:
        train_sizes: Training set sizes
        train_scores: Training scores
        val_scores: Validation scores
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    plt.plot(train_sizes, train_scores, 'o-', label='Training Score', linewidth=2)
    plt.plot(train_sizes, val_scores, 's-', label='Validation Score', linewidth=2)
    
    plt.xlabel('Training Set Size')
    plt.ylabel('Score')
    plt.title('Learning Curves')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Learning curves plot saved to {save_path}")
    
    plt.show()
