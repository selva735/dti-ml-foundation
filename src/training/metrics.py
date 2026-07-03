"""Evaluation metrics and early-stopping for DTI training.

Supported regression metrics:
  - MSE, RMSE, MAE
  - Pearson correlation coefficient (R)
  - Spearman rank correlation (ρ)
  - Concordance Index (CI) – standard in drug-affinity benchmarks

Classification metrics (for binary interaction labels):
  - ROC-AUC, PR-AUC, Accuracy, F1
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np
import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------------

def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return math.sqrt(_mse(y_true, y_pred))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.std() < 1e-8 or y_pred.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from scipy.stats import spearmanr  # optional dependency

    corr, _ = spearmanr(y_true, y_pred)
    return float(corr) if not math.isnan(corr) else 0.0


def _concordance_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the concordance index (CI / Harrell's C-index).

    CI counts the fraction of pairs (i, j) with y_true[i] > y_true[j] for
    which y_pred[i] also > y_pred[j].
    """
    n = len(y_true)
    concordant = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            if y_true[i] != y_true[j]:
                total += 1
                if (y_true[i] > y_true[j]) == (y_pred[i] > y_pred[j]):
                    concordant += 1
    return concordant / total if total > 0 else 0.5


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-8)


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def _roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")


def _pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        from sklearn.metrics import average_precision_score

        return float(average_precision_score(y_true, y_score))
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: Tensor | np.ndarray,
    y_pred: Tensor | np.ndarray,
    task: str = "regression",
    fast: bool = False,
) -> Dict[str, float]:
    """Compute evaluation metrics for DTI prediction.

    Args:
        y_true: Ground-truth labels as a 1-D array or tensor.
        y_pred: Predicted values as a 1-D array or tensor.
        task: Either ``"regression"`` or ``"classification"``.
        fast: If True, skip the O(N²) concordance index (useful for large
            validation sets during training).

    Returns:
        Dict mapping metric names to scalar floats.
    """
    if isinstance(y_true, Tensor):
        y_true = y_true.detach().cpu().numpy().flatten()
    if isinstance(y_pred, Tensor):
        y_pred = y_pred.detach().cpu().numpy().flatten()

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if task == "regression":
        metrics = {
            "mse": _mse(y_true, y_pred),
            "rmse": _rmse(y_true, y_pred),
            "mae": _mae(y_true, y_pred),
            "pearson": _pearson(y_true, y_pred),
            "r2": _r2_score(y_true, y_pred),
        }
        try:
            metrics["spearman"] = _spearman(y_true, y_pred)
        except ImportError:
            pass
        if not fast:
            metrics["ci"] = _concordance_index(y_true, y_pred)
        return metrics

    if task == "classification":
        threshold = 0.5
        y_binary = (y_pred >= threshold).astype(int)
        correct = (y_binary == y_true.astype(int)).sum()
        return {
            "accuracy": float(correct / len(y_true)),
            "roc_auc": _roc_auc(y_true, y_pred),
            "pr_auc": _pr_auc(y_true, y_pred),
        }

    raise ValueError(f"Unknown task: {task!r}. Use 'regression' or 'classification'.")


class EarlyStopping:
    """Monitor a metric and stop training when it stops improving.

    Args:
        patience: Number of epochs without improvement before stopping.
        min_delta: Minimum absolute change to count as an improvement.
        mode: ``"min"`` for loss-like metrics, ``"max"`` for score-like metrics.
        restore_best: If True, restore the best model state when stopping.
    """

    def __init__(
        self,
        patience: int = 20,
        min_delta: float = 1e-4,
        mode: str = "min",
        restore_best: bool = True,
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best
        self._counter = 0
        self._best_score: float | None = None
        self.best_state: dict | None = None
        self.stopped = False

    def _is_better(self, score: float) -> bool:
        if self._best_score is None:
            return True
        if self.mode == "min":
            return score < self._best_score - self.min_delta
        return score > self._best_score + self.min_delta

    def step(self, score: float, model=None) -> bool:
        """Update state with the new score.

        Args:
            score: Current epoch metric value.
            model: If provided and *restore_best* is True, saves the model
                state dict when a new best is found.

        Returns:
            ``True`` if training should stop, ``False`` otherwise.
        """
        if self._is_better(score):
            self._best_score = score
            self._counter = 0
            if self.restore_best and model is not None:
                import copy
                self.best_state = copy.deepcopy(model.state_dict())
        else:
            self._counter += 1

        if self._counter >= self.patience:
            self.stopped = True
            return True
        return False

    def restore(self, model) -> None:
        """Restore model to best observed state (if restore_best=True)."""
        if self.restore_best and self.best_state is not None:
            model.load_state_dict(self.best_state)
