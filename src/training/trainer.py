"""Training loop for CMAT-DTI.

Features:
  - Learning-rate scheduling (ReduceLROnPlateau or cosine)
  - Gradient clipping
  - Mixed-precision training (AMP) when a GPU is available
  - TensorBoard logging (optional)
  - Checkpoint save/load
  - Integration with :class:`EarlyStopping`
"""

from __future__ import annotations

import os
import time
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

from .metrics import EarlyStopping, compute_metrics


class Trainer:
    """Manages the CMAT-DTI training/validation lifecycle.

    Args:
        model: The :class:`CMATDTI` model to train.
        train_loader: DataLoader for the training set.
        val_loader: DataLoader for the validation set.
        optimizer: PyTorch optimiser.
        criterion: Loss function (e.g. ``nn.MSELoss()``).
        device: Torch device string (``"cpu"``, ``"cuda"``, ``"mps"``).
        task: ``"regression"`` or ``"classification"``.
        scheduler_type: ``"plateau"`` or ``"cosine"``.
        max_grad_norm: Gradient clipping norm (0 = disabled).
        use_amp: Enable automatic mixed precision (GPU only).
        checkpoint_dir: Directory to save best checkpoint.
        log_dir: Directory for TensorBoard logs (None = disabled).
        early_stopping_patience: Epochs without improvement before stopping.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: str = "cpu",
        task: str = "regression",
        scheduler_type: str = "plateau",
        max_grad_norm: float = 1.0,
        use_amp: bool = False,
        checkpoint_dir: str = "checkpoints",
        log_dir: Optional[str] = None,
        early_stopping_patience: int = 20,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = torch.device(device)
        self.task = task
        self.max_grad_norm = max_grad_norm
        self.use_amp = use_amp and self.device.type == "cuda"
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.scaler = GradScaler() if self.use_amp else None

        # Learning-rate scheduler
        if scheduler_type == "cosine":
            self.scheduler = CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-6)
        else:
            self.scheduler = ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
            )
        self.scheduler_type = scheduler_type

        # Early stopping (minimize validation loss)
        self.early_stopping = EarlyStopping(
            patience=early_stopping_patience, mode="min", restore_best=True
        )

        # Optional TensorBoard
        self.writer = None
        if log_dir is not None:
            try:
                from torch.utils.tensorboard import SummaryWriter  # type: ignore

                self.writer = SummaryWriter(log_dir)
            except ImportError:
                print("[Trainer] TensorBoard not available; logging disabled.")

        self.model.to(self.device)
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_pearson": [],
        }

    # ------------------------------------------------------------------
    # Core training methods
    # ------------------------------------------------------------------

    def _batch_to_device(self, batch: dict) -> dict:
        return {
            k: v.to(self.device) if isinstance(v, Tensor) else v
            for k, v in batch.items()
        }

    def _forward(self, batch: dict) -> Tensor:
        """Run a forward pass and return model predictions."""
        b = self._batch_to_device(batch)
        return self.model(
            atom_features=b["atom_features"],
            dist_matrix=b["dist_matrix"],
            protein_tokens=b.get("tokens"),
            atom_mask=b.get("atom_mask"),
            protein_padding_mask=b.get("protein_padding_mask"),
        ).squeeze(-1)

    def train_epoch(self) -> float:
        """Train for one epoch.

        Returns:
            Mean training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        n_batches = len(self.train_loader)

        for batch in self.train_loader:
            self.optimizer.zero_grad()
            labels = batch["label"].to(self.device)

            if self.use_amp:
                with autocast():
                    preds = self._forward(batch)
                    loss = self.criterion(preds, labels)
                self.scaler.scale(loss).backward()  # type: ignore[union-attr]
                if self.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                preds = self._forward(batch)
                loss = self.criterion(preds, labels)
                loss.backward()
                if self.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

            total_loss += loss.item()

        return total_loss / n_batches

    @torch.no_grad()
    def evaluate(self, loader: Optional[DataLoader] = None, fast: bool = True) -> Dict[str, float]:
        """Evaluate the model on *loader*.

        Args:
            loader: DataLoader to evaluate on (defaults to ``val_loader``).
            fast: If True, skip concordance index computation.

        Returns:
            Dict of metric name → value including ``"loss"``.
        """
        self.model.eval()
        if loader is None:
            loader = self.val_loader

        all_preds: list[Tensor] = []
        all_labels: list[Tensor] = []
        total_loss = 0.0

        for batch in loader:
            labels = batch["label"].to(self.device)
            preds = self._forward(batch)
            loss = self.criterion(preds, labels)
            total_loss += loss.item()
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

        y_pred = torch.cat(all_preds)
        y_true = torch.cat(all_labels)
        metrics = compute_metrics(y_true, y_pred, task=self.task, fast=fast)
        metrics["loss"] = total_loss / len(loader)
        return metrics

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(
        self,
        n_epochs: int = 100,
        eval_every: int = 1,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """Run the full training loop.

        Args:
            n_epochs: Maximum number of epochs.
            eval_every: Evaluate on validation set every *eval_every* epochs.
            verbose: Print progress to stdout.

        Returns:
            Training history dict.
        """
        best_val_loss = float("inf")

        for epoch in range(1, n_epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch()

            val_metrics: Dict[str, float] = {}
            if epoch % eval_every == 0:
                val_metrics = self.evaluate()
                val_loss = val_metrics["loss"]
                val_pearson = val_metrics.get("pearson", 0.0)

                self.history["train_loss"].append(train_loss)
                self.history["val_loss"].append(val_loss)
                self.history["val_pearson"].append(val_pearson)

                # Update LR scheduler
                if self.scheduler_type == "plateau":
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

                # Save best checkpoint
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._save_checkpoint("best_model.pt")

                # TensorBoard
                if self.writer is not None:
                    self.writer.add_scalar("Loss/train", train_loss, epoch)
                    self.writer.add_scalar("Loss/val", val_loss, epoch)
                    for k, v in val_metrics.items():
                        self.writer.add_scalar(f"Metrics/{k}", v, epoch)

                elapsed = time.time() - t0
                if verbose:
                    msg = (
                        f"Epoch {epoch:4d}/{n_epochs} | "
                        f"train_loss={train_loss:.4f} | "
                        f"val_loss={val_loss:.4f} | "
                        f"pearson={val_pearson:.4f} | "
                        f"lr={self._current_lr():.2e} | "
                        f"time={elapsed:.1f}s"
                    )
                    print(msg)

                # Early stopping
                if self.early_stopping.step(val_loss, self.model):
                    if verbose:
                        print(f"Early stopping at epoch {epoch}.")
                    break

        # Restore best model weights
        self.early_stopping.restore(self.model)
        if self.writer is not None:
            self.writer.close()
        return self.history

    # ------------------------------------------------------------------
    # Checkpoint utilities
    # ------------------------------------------------------------------

    def _current_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def _save_checkpoint(self, filename: str) -> None:
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "history": self.history,
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        """Load model and optimizer state from a checkpoint file.

        Args:
            path: Path to the checkpoint ``.pt`` file.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint.get("history", self.history)
        print(f"Loaded checkpoint from {path}")
