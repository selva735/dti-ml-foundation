"""Tests for training modules."""

import pytest
import sys
from pathlib import Path
import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from training.evaluator import DTIEvaluator
from training.trainer import create_optimizer, create_scheduler


class TestDTIEvaluator:
    """Tests for DTI evaluator."""
    
    def test_regression_metrics(self):
        """Test computation of regression metrics."""
        evaluator = DTIEvaluator(task_type='regression')
        
        # Perfect predictions
        targets = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predictions = targets.copy()
        
        metrics = evaluator.compute_metrics(predictions, targets)
        
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'pearson' in metrics
        
        # Perfect predictions should have zero error
        assert metrics['mse'] == pytest.approx(0.0, abs=1e-6)
        assert metrics['pearson'] == pytest.approx(1.0, abs=1e-6)
    
    def test_metrics_with_noise(self):
        """Test metrics with noisy predictions."""
        evaluator = DTIEvaluator(task_type='regression')
        
        np.random.seed(42)
        targets = np.random.uniform(4, 10, 100)
        noise = np.random.normal(0, 0.5, 100)
        predictions = targets + noise
        
        metrics = evaluator.compute_metrics(predictions, targets)
        
        # Should have some error but good correlation
        assert metrics['mse'] > 0
        assert metrics['pearson'] > 0.8  # High correlation despite noise


class TestOptimizerScheduler:
    """Tests for optimizer and scheduler creation."""
    
    def test_create_adam_optimizer(self):
        """Test Adam optimizer creation."""
        model = torch.nn.Linear(10, 1)
        
        optimizer = create_optimizer(
            model,
            optimizer_name='adam',
            learning_rate=0.001,
        )
        
        assert isinstance(optimizer, torch.optim.Adam)
        assert optimizer.param_groups[0]['lr'] == 0.001
    
    def test_create_cosine_scheduler(self):
        """Test cosine scheduler creation."""
        model = torch.nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        scheduler = create_scheduler(
            optimizer,
            scheduler_name='cosine',
            n_epochs=100,
        )
        
        assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
    
    def test_create_plateau_scheduler(self):
        """Test plateau scheduler creation."""
        model = torch.nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        scheduler = create_scheduler(
            optimizer,
            scheduler_name='plateau',
            n_epochs=100,
            patience=10,
        )
        
        assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)


class TestMetricsComputation:
    """Tests for metrics computation."""
    
    def test_pearson_correlation(self):
        """Test Pearson correlation computation."""
        evaluator = DTIEvaluator(task_type='regression')
        
        # Perfect positive correlation
        targets = np.array([1, 2, 3, 4, 5], dtype=float)
        predictions = targets * 2  # Scaled version
        
        metrics = evaluator.compute_metrics(predictions, targets)
        
        # Pearson should be 1 for perfect linear relationship
        assert metrics['pearson'] == pytest.approx(1.0, abs=1e-6)
    
    def test_r2_score(self):
        """Test R² score computation."""
        evaluator = DTIEvaluator(task_type='regression')
        
        # Perfect predictions
        targets = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predictions = targets.copy()
        
        metrics = evaluator.compute_metrics(predictions, targets)
        
        # Perfect predictions should have R² = 1
        assert metrics['r2'] == pytest.approx(1.0, abs=1e-6)
