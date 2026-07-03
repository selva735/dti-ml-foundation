"""Unit tests for training utilities (metrics, EarlyStopping, Trainer).

Trainer tests use a tiny model and in-memory dataloader so they run quickly
without GPU or large datasets.
"""

import math
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.metrics import compute_metrics, EarlyStopping
from src.models.cmat_dti import CMATDTI, ModelConfig


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_perfect_regression(self):
        y = torch.arange(10, dtype=torch.float)
        metrics = compute_metrics(y, y, task="regression", fast=True)
        assert metrics["mse"] == pytest.approx(0.0, abs=1e-6)
        assert metrics["rmse"] == pytest.approx(0.0, abs=1e-6)
        assert metrics["pearson"] == pytest.approx(1.0, abs=1e-6)

    def test_regression_returns_expected_keys(self):
        y = torch.rand(20)
        yhat = torch.rand(20)
        metrics = compute_metrics(y, yhat, task="regression", fast=True)
        for key in ["mse", "rmse", "mae", "pearson", "r2"]:
            assert key in metrics, f"Missing key: {key}"

    def test_concordance_index_perfect(self):
        import numpy as np
        y = np.array([1.0, 2.0, 3.0, 4.0])
        metrics = compute_metrics(y, y, task="regression", fast=False)
        assert metrics["ci"] == pytest.approx(1.0, abs=1e-6)

    def test_classification_keys(self):
        y = torch.tensor([0, 1, 0, 1, 1], dtype=torch.float)
        yhat = torch.tensor([0.1, 0.9, 0.2, 0.8, 0.7])
        metrics = compute_metrics(y, yhat, task="classification")
        for key in ["accuracy", "roc_auc", "pr_auc"]:
            assert key in metrics

    def test_unknown_task_raises(self):
        with pytest.raises(ValueError):
            compute_metrics([1.0], [1.0], task="unknown")

    def test_accepts_numpy_arrays(self):
        import numpy as np
        y = np.array([1.0, 2.0, 3.0])
        metrics = compute_metrics(y, y, task="regression", fast=True)
        assert metrics["mse"] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# EarlyStopping tests
# ---------------------------------------------------------------------------

class TestEarlyStopping:
    def test_stops_after_patience(self):
        es = EarlyStopping(patience=3, mode="min")
        scores = [1.0, 1.0, 1.0, 1.0]  # no improvement
        results = [es.step(s) for s in scores]
        assert results[-1] is True, "Should stop after patience exceeded"

    def test_resets_counter_on_improvement(self):
        es = EarlyStopping(patience=3, mode="min")
        assert not es.step(1.0)
        assert not es.step(1.0)
        assert not es.step(0.5)  # improvement → counter resets
        assert not es.step(0.5)
        assert not es.step(0.5)
        assert es.step(0.5)  # now 3 non-improving steps after last best

    def test_max_mode(self):
        es = EarlyStopping(patience=2, mode="max")
        es.step(0.5)
        es.step(0.5)
        assert es.step(0.5)  # stopped

    def test_restore_best(self):
        model = nn.Linear(2, 1)
        es = EarlyStopping(patience=5, mode="min", restore_best=True)
        es.step(1.0, model)  # best = 1.0
        # Make model weights change
        original_weight = model.weight.data.clone()
        with torch.no_grad():
            model.weight.fill_(99.0)
        es.step(0.5, model)  # new best
        best_weight = model.weight.data.clone()
        # Change weights again and restore
        with torch.no_grad():
            model.weight.fill_(77.0)
        es.restore(model)
        assert torch.allclose(model.weight.data, best_weight)

    def test_no_stop_before_patience(self):
        es = EarlyStopping(patience=5, mode="min")
        for i in range(4):
            assert not es.step(1.0)


# ---------------------------------------------------------------------------
# Trainer integration test (tiny model, in-memory data)
# ---------------------------------------------------------------------------

def _make_tiny_model():
    cfg = ModelConfig(
        atom_feat_dim=57,
        drug_embed_dim=16,
        drug_num_layers=1,
        drug_num_heads=2,
        drug_ffn_dim=32,
        protein_embed_dim=16,
        protein_num_layers=1,
        protein_num_heads=2,
        protein_ffn_dim=32,
        cross_dim=16,
        cross_num_heads=2,
        cross_num_layers=1,
        fusion_output_dim=32,
        head_hidden_dims=[16],
        output_dim=1,
        mc_dropout_samples=3,
    )
    return CMATDTI(cfg)


def _make_tiny_loader(n=8, batch_size=4):
    """Create a tiny DataLoader returning properly structured batches."""
    from src.data.dataset import collate_fn

    # Build fake dataset items
    items = []
    for _ in range(n):
        n_atoms = 10
        seq_len = 12
        items.append({
            "drug_id": "d0",
            "target_id": "t0",
            "atom_features": torch.rand(n_atoms, 57),
            "dist_matrix": torch.randint(0, 5, (n_atoms, n_atoms)),
            "atom_mask": torch.zeros(n_atoms, dtype=torch.bool),
            "tokens": torch.randint(1, 20, (seq_len,)),
            "padding_mask": torch.zeros(seq_len, dtype=torch.bool),
            "label": torch.tensor(5.0),
        })

    loader = DataLoader(items, batch_size=batch_size, collate_fn=collate_fn)
    return loader


class TestTrainer:
    def test_train_runs_without_error(self, tmp_path):
        from src.training.trainer import Trainer

        model = _make_tiny_model()
        loader = _make_tiny_loader()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        trainer = Trainer(
            model=model,
            train_loader=loader,
            val_loader=loader,
            optimizer=optimizer,
            criterion=criterion,
            device="cpu",
            task="regression",
            early_stopping_patience=2,
            checkpoint_dir=str(tmp_path),
        )
        history = trainer.train(n_epochs=3, verbose=False)
        assert "train_loss" in history
        assert len(history["train_loss"]) == 3

    def test_evaluate_returns_metrics(self):
        from src.training.trainer import Trainer

        model = _make_tiny_model()
        loader = _make_tiny_loader()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        trainer = Trainer(
            model=model,
            train_loader=loader,
            val_loader=loader,
            optimizer=optimizer,
            criterion=criterion,
            device="cpu",
            checkpoint_dir="/tmp/cmat_test_ckpt",
        )
        metrics = trainer.evaluate(fast=True)
        assert "loss" in metrics
        assert "mse" in metrics

    def test_checkpoint_saved(self, tmp_path):
        from src.training.trainer import Trainer
        import os

        model = _make_tiny_model()
        loader = _make_tiny_loader()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        trainer = Trainer(
            model=model,
            train_loader=loader,
            val_loader=loader,
            optimizer=optimizer,
            criterion=criterion,
            device="cpu",
            early_stopping_patience=2,
            checkpoint_dir=str(tmp_path),
        )
        trainer.train(n_epochs=2, verbose=False)
        assert os.path.exists(os.path.join(str(tmp_path), "best_model.pt"))
