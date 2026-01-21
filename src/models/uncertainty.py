"""Uncertainty estimation for DTI predictions.

This module implements uncertainty quantification methods including
Monte Carlo Dropout and ensemble approaches.
"""

from typing import List, Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


class MCDropout(nn.Module):
    """Monte Carlo Dropout for uncertainty estimation.
    
    Applies dropout at inference time to estimate epistemic uncertainty.
    """
    
    def __init__(self, model: nn.Module, n_samples: int = 30):
        """Initialize MC Dropout.
        
        Args:
            model: Base model to apply MC Dropout on
            n_samples: Number of forward passes for uncertainty estimation
        """
        super().__init__()
        self.model = model
        self.n_samples = n_samples
    
    def enable_dropout(self):
        """Enable dropout layers during inference."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    
    def forward(self, *args, **kwargs):
        """Standard forward pass."""
        return self.model(*args, **kwargs)
    
    def predict_with_uncertainty(
        self,
        drug_graph,
        protein_emb: torch.Tensor,
        return_all_samples: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Predict with uncertainty estimation.
        
        Args:
            drug_graph: PyTorch Geometric Batch object
            protein_emb: Protein embeddings
            return_all_samples: Whether to return all MC samples
            
        Returns:
            Dictionary with predictions and uncertainty estimates
        """
        self.enable_dropout()
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(self.n_samples):
                pred, _ = self.model(drug_graph, protein_emb)
                predictions.append(pred)
        
        # Stack predictions
        predictions = torch.stack(predictions, dim=0)  # [n_samples, batch_size, 1]
        
        # Compute statistics
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        
        # Epistemic uncertainty (model uncertainty)
        epistemic_uncertainty = std_pred
        
        result = {
            'mean': mean_pred,
            'std': std_pred,
            'epistemic_uncertainty': epistemic_uncertainty,
        }
        
        if return_all_samples:
            result['all_samples'] = predictions
        
        return result


class EnsembleModel(nn.Module):
    """Ensemble of models for uncertainty estimation.
    
    Combines predictions from multiple independently trained models.
    """
    
    def __init__(self, models: List[nn.Module]):
        """Initialize ensemble.
        
        Args:
            models: List of trained models
        """
        super().__init__()
        self.models = nn.ModuleList(models)
        self.n_models = len(models)
    
    def forward(self, *args, **kwargs):
        """Forward pass through all models."""
        predictions = []
        
        for model in self.models:
            pred, _ = model(*args, **kwargs)
            predictions.append(pred)
        
        # Average predictions
        mean_pred = torch.stack(predictions, dim=0).mean(dim=0)
        
        return mean_pred, None
    
    def predict_with_uncertainty(
        self,
        drug_graph,
        protein_emb: torch.Tensor,
        return_all_predictions: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Predict with uncertainty from ensemble.
        
        Args:
            drug_graph: PyTorch Geometric Batch object
            protein_emb: Protein embeddings
            return_all_predictions: Whether to return all model predictions
            
        Returns:
            Dictionary with predictions and uncertainty estimates
        """
        predictions = []
        
        with torch.no_grad():
            for model in self.models:
                pred, _ = model(drug_graph, protein_emb)
                predictions.append(pred)
        
        # Stack predictions
        predictions = torch.stack(predictions, dim=0)  # [n_models, batch_size, 1]
        
        # Compute statistics
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        
        # Model disagreement as epistemic uncertainty
        epistemic_uncertainty = std_pred
        
        result = {
            'mean': mean_pred,
            'std': std_pred,
            'epistemic_uncertainty': epistemic_uncertainty,
        }
        
        if return_all_predictions:
            result['all_predictions'] = predictions
        
        return result


class UncertaintyEstimator:
    """Unified interface for uncertainty estimation.
    
    Provides methods for different uncertainty estimation approaches.
    """
    
    def __init__(
        self,
        model: nn.Module,
        method: str = "mcdropout",
        n_samples: int = 30,
        ensemble_models: List[nn.Module] = None,
    ):
        """Initialize uncertainty estimator.
        
        Args:
            model: Base model for uncertainty estimation
            method: Estimation method ('mcdropout' or 'ensemble')
            n_samples: Number of samples for MC Dropout
            ensemble_models: List of models for ensemble (if method='ensemble')
        """
        self.method = method
        
        if method == "mcdropout":
            self.estimator = MCDropout(model, n_samples=n_samples)
        elif method == "ensemble":
            if ensemble_models is None:
                raise ValueError("ensemble_models required for ensemble method")
            self.estimator = EnsembleModel(ensemble_models)
        else:
            raise ValueError(f"Unknown uncertainty method: {method}")
    
    def predict(
        self,
        drug_graph,
        protein_emb: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Make predictions with uncertainty estimates.
        
        Args:
            drug_graph: PyTorch Geometric Batch object
            protein_emb: Protein embeddings
            
        Returns:
            Dictionary with predictions and uncertainties
        """
        return self.estimator.predict_with_uncertainty(drug_graph, protein_emb)
    
    def get_confidence_intervals(
        self,
        predictions: Dict[str, torch.Tensor],
        confidence_level: float = 0.95,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute confidence intervals.
        
        Args:
            predictions: Dictionary from predict_with_uncertainty
            confidence_level: Confidence level (e.g., 0.95 for 95% CI)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        mean = predictions['mean']
        std = predictions['std']
        
        # Assuming normal distribution
        z_score = {
            0.90: 1.645,
            0.95: 1.96,
            0.99: 2.576,
        }.get(confidence_level, 1.96)
        
        margin = z_score * std
        lower = mean - margin
        upper = mean + margin
        
        return lower, upper


def calibrate_uncertainty(
    estimator: UncertaintyEstimator,
    val_loader,
    device: str = 'cuda',
) -> Dict[str, float]:
    """Calibrate uncertainty estimates on validation set.
    
    Computes calibration metrics to assess uncertainty quality.
    
    Args:
        estimator: Uncertainty estimator
        val_loader: Validation data loader
        device: Device to run on
        
    Returns:
        Dictionary with calibration metrics
    """
    all_predictions = []
    all_uncertainties = []
    all_targets = []
    
    estimator.estimator.model.eval()
    
    for batch in tqdm(val_loader, desc="Calibrating uncertainty"):
        drug_graph = batch['drug_graph'].to(device)
        protein_emb = batch['protein_emb'].to(device)
        affinity = batch['affinity'].to(device)
        
        # Get predictions with uncertainty
        result = estimator.predict(drug_graph, protein_emb)
        
        all_predictions.append(result['mean'].cpu())
        all_uncertainties.append(result['std'].cpu())
        all_targets.append(affinity.cpu())
    
    predictions = torch.cat(all_predictions, dim=0).numpy()
    uncertainties = torch.cat(all_uncertainties, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()
    
    # Compute calibration metrics
    errors = np.abs(predictions - targets)
    
    # Correlation between uncertainty and error
    correlation = np.corrcoef(uncertainties.flatten(), errors.flatten())[0, 1]
    
    # Expected calibration error (simplified version)
    # Sort by uncertainty
    sorted_indices = np.argsort(uncertainties.flatten())
    sorted_uncertainties = uncertainties.flatten()[sorted_indices]
    sorted_errors = errors.flatten()[sorted_indices]
    
    # Divide into bins
    n_bins = 10
    bin_size = len(sorted_errors) // n_bins
    
    calibration_errors = []
    for i in range(n_bins):
        start_idx = i * bin_size
        end_idx = (i + 1) * bin_size if i < n_bins - 1 else len(sorted_errors)
        
        bin_uncertainties = sorted_uncertainties[start_idx:end_idx]
        bin_errors = sorted_errors[start_idx:end_idx]
        
        expected_error = bin_uncertainties.mean()
        observed_error = bin_errors.mean()
        
        calibration_errors.append(abs(expected_error - observed_error))
    
    expected_calibration_error = np.mean(calibration_errors)
    
    return {
        'uncertainty_error_correlation': correlation,
        'expected_calibration_error': expected_calibration_error,
        'mean_uncertainty': uncertainties.mean(),
        'mean_error': errors.mean(),
    }
