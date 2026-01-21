"""Evaluation metrics for DTI prediction.

This module provides comprehensive evaluation metrics for assessing model
performance on drug-target interaction prediction tasks.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    average_precision_score,
)
from scipy.stats import pearsonr, spearmanr


class DTIEvaluator:
    """Evaluator for DTI prediction models.
    
    Computes various regression and classification metrics for model evaluation.
    """
    
    def __init__(self, task_type: str = "regression"):
        """Initialize evaluator.
        
        Args:
            task_type: Type of task ('regression' or 'classification')
        """
        self.task_type = task_type
    
    def compute_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        return_all: bool = True,
    ) -> Dict[str, float]:
        """Compute evaluation metrics.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            return_all: Whether to return all metrics
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        if self.task_type == "regression":
            # Regression metrics
            metrics['mse'] = mean_squared_error(targets, predictions)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(targets, predictions)
            
            # Correlation metrics
            if len(predictions) > 1:
                pearson_corr, pearson_p = pearsonr(predictions.flatten(), targets.flatten())
                spearman_corr, spearman_p = spearmanr(predictions.flatten(), targets.flatten())
                
                metrics['pearson'] = pearson_corr
                metrics['pearson_p'] = pearson_p
                metrics['spearman'] = spearman_corr
                metrics['spearman_p'] = spearman_p
            
            # R² score
            metrics['r2'] = r2_score(targets, predictions)
        
        elif self.task_type == "classification":
            # Classification metrics
            binary_preds = (predictions >= 0.5).astype(int)
            binary_targets = targets.astype(int)
            
            # Accuracy
            metrics['accuracy'] = (binary_preds == binary_targets).mean()
            
            # ROC-AUC
            try:
                metrics['roc_auc'] = roc_auc_score(binary_targets, predictions)
            except ValueError:
                metrics['roc_auc'] = 0.0
            
            # PR-AUC
            try:
                metrics['pr_auc'] = average_precision_score(binary_targets, predictions)
            except ValueError:
                metrics['pr_auc'] = 0.0
        
        if not return_all:
            # Return only primary metrics
            if self.task_type == "regression":
                metrics = {
                    'mse': metrics['mse'],
                    'rmse': metrics['rmse'],
                    'pearson': metrics.get('pearson', 0.0),
                }
            else:
                metrics = {
                    'accuracy': metrics['accuracy'],
                    'roc_auc': metrics['roc_auc'],
                }
        
        return metrics
    
    def evaluate_model(
        self,
        model,
        data_loader,
        device: str = 'cuda',
        return_predictions: bool = False,
    ) -> Dict[str, float]:
        """Evaluate model on a dataset.
        
        Args:
            model: Model to evaluate
            data_loader: DataLoader for evaluation data
            device: Device to run on
            return_predictions: Whether to return predictions
            
        Returns:
            Dictionary of metrics (and predictions if requested)
        """
        model.eval()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in data_loader:
                drug_graph = batch['drug_graph'].to(device)
                protein_emb = batch['protein_emb'].to(device)
                affinity = batch['affinity'].to(device)
                
                # Forward pass
                predictions, _ = model(drug_graph, protein_emb)
                
                all_predictions.append(predictions.cpu())
                all_targets.append(affinity.cpu())
        
        # Concatenate all batches
        predictions = torch.cat(all_predictions, dim=0).numpy()
        targets = torch.cat(all_targets, dim=0).numpy()
        
        # Compute metrics
        metrics = self.compute_metrics(predictions, targets)
        
        if return_predictions:
            metrics['predictions'] = predictions
            metrics['targets'] = targets
        
        return metrics


class StratifiedEvaluator:
    """Evaluator with stratified analysis.
    
    Evaluates model performance across different strata (e.g., by drug similarity,
    target family, affinity range).
    """
    
    def __init__(self, base_evaluator: DTIEvaluator):
        """Initialize stratified evaluator.
        
        Args:
            base_evaluator: Base evaluator for computing metrics
        """
        self.base_evaluator = base_evaluator
    
    def evaluate_by_strata(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        strata_labels: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate by different strata.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            strata_labels: Stratum label for each sample
            
        Returns:
            Dictionary mapping stratum to metrics
        """
        unique_strata = np.unique(strata_labels)
        results = {}
        
        for stratum in unique_strata:
            mask = strata_labels == stratum
            
            if mask.sum() == 0:
                continue
            
            strata_preds = predictions[mask]
            strata_targets = targets[mask]
            
            metrics = self.base_evaluator.compute_metrics(strata_preds, strata_targets)
            results[str(stratum)] = metrics
        
        return results
    
    def evaluate_by_affinity_range(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        n_bins: int = 5,
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate by affinity value ranges.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            n_bins: Number of bins to divide affinity range
            
        Returns:
            Dictionary mapping range to metrics
        """
        # Create bins
        bins = np.linspace(targets.min(), targets.max(), n_bins + 1)
        bin_labels = np.digitize(targets.flatten(), bins) - 1
        bin_labels = np.clip(bin_labels, 0, n_bins - 1)
        
        results = {}
        
        for i in range(n_bins):
            mask = bin_labels == i
            
            if mask.sum() == 0:
                continue
            
            bin_preds = predictions[mask]
            bin_targets = targets[mask]
            
            metrics = self.base_evaluator.compute_metrics(bin_preds, bin_targets)
            range_str = f"[{bins[i]:.2f}, {bins[i+1]:.2f}]"
            results[range_str] = metrics
            results[range_str]['n_samples'] = mask.sum()
        
        return results


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    """Print metrics in a formatted way.
    
    Args:
        metrics: Dictionary of metrics
        prefix: Prefix for metric names
    """
    print(f"\n{prefix}Evaluation Metrics:")
    print("=" * 50)
    
    for name, value in metrics.items():
        if isinstance(value, (int, float, np.number)):
            print(f"{name:20s}: {value:.4f}")
        else:
            print(f"{name:20s}: {value}")
    
    print("=" * 50)


def compare_models(
    results_dict: Dict[str, Dict[str, float]],
    metric: str = 'rmse',
) -> Dict[str, float]:
    """Compare multiple models on a specific metric.
    
    Args:
        results_dict: Dictionary mapping model name to metrics
        metric: Metric to compare
        
    Returns:
        Dictionary with comparison results
    """
    comparison = {}
    
    for model_name, metrics in results_dict.items():
        if metric in metrics:
            comparison[model_name] = metrics[metric]
    
    # Sort by metric (lower is better for error metrics)
    if metric in ['mse', 'rmse', 'mae']:
        comparison = dict(sorted(comparison.items(), key=lambda x: x[1]))
    else:
        # Higher is better for correlation and R²
        comparison = dict(sorted(comparison.items(), key=lambda x: x[1], reverse=True))
    
    print(f"\nModel Comparison ({metric}):")
    print("=" * 50)
    for i, (model_name, value) in enumerate(comparison.items(), 1):
        print(f"{i}. {model_name:30s}: {value:.4f}")
    print("=" * 50)
    
    return comparison
