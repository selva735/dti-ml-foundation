"""Training loop and utilities for DTI prediction models.

This module provides the main training infrastructure including training loops,
learning rate scheduling, early stopping, and checkpoint management.
"""

from typing import Dict, List, Optional, Tuple, Callable
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import json


class Trainer:
    """Trainer for DTI prediction models.
    
    Handles training loop, validation, checkpointing, and logging.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str = 'cuda',
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        evaluator = None,
        checkpoint_dir: str = './checkpoints',
        log_dir: str = './runs',
        gradient_clip: Optional[float] = None,
        mixed_precision: bool = False,
    ):
        """Initialize trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            criterion: Loss function
            device: Device to train on
            scheduler: Learning rate scheduler (optional)
            evaluator: Evaluator for computing metrics (optional)
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory for tensorboard logs
            gradient_clip: Gradient clipping value (optional)
            mixed_precision: Whether to use mixed precision training
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.evaluator = evaluator
        self.gradient_clip = gradient_clip
        self.mixed_precision = mixed_precision
        
        # Create directories
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        # Mixed precision scaler
        if mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_metrics': [],
        }
        
        self.best_val_loss = float('inf')
        self.best_epoch = 0
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            drug_graph = batch['drug_graph'].to(self.device)
            protein_emb = batch['protein_emb'].to(self.device)
            affinity = batch['affinity'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    predictions, _ = self.model(drug_graph, protein_emb)
                    loss = self.criterion(predictions, affinity)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.gradient_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip
                    )
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            
            else:
                # Standard training
                predictions, _ = self.model(drug_graph, protein_emb)
                loss = self.criterion(predictions, affinity)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip
                    )
                
                # Optimizer step
                self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            n_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
            
            # Log to tensorboard
            global_step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('train/batch_loss', loss.item(), global_step)
        
        avg_loss = total_loss / n_batches
        return avg_loss
    
    def validate(self, epoch: int) -> Tuple[float, Dict]:
        """Validate model.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Tuple of (validation loss, metrics dictionary)
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                drug_graph = batch['drug_graph'].to(self.device)
                protein_emb = batch['protein_emb'].to(self.device)
                affinity = batch['affinity'].to(self.device)
                
                # Forward pass
                predictions, _ = self.model(drug_graph, protein_emb)
                loss = self.criterion(predictions, affinity)
                
                total_loss += loss.item()
                n_batches += 1
                
                # Collect predictions
                all_predictions.append(predictions.cpu())
                all_targets.append(affinity.cpu())
        
        avg_loss = total_loss / n_batches
        
        # Compute metrics
        metrics = {}
        if self.evaluator is not None:
            predictions = torch.cat(all_predictions, dim=0).numpy()
            targets = torch.cat(all_targets, dim=0).numpy()
            metrics = self.evaluator.compute_metrics(predictions, targets)
        
        return avg_loss, metrics
    
    def train(
        self,
        n_epochs: int,
        early_stopping_patience: Optional[int] = None,
        save_best: bool = True,
        save_every: Optional[int] = None,
    ) -> Dict:
        """Train model for multiple epochs.
        
        Args:
            n_epochs: Number of epochs to train
            early_stopping_patience: Patience for early stopping (optional)
            save_best: Whether to save best model
            save_every: Save checkpoint every N epochs (optional)
            
        Returns:
            Training history dictionary
        """
        print(f"\nStarting training for {n_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        
        patience_counter = 0
        
        for epoch in range(1, n_epochs + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{n_epochs}")
            print(f"{'='*60}")
            
            # Train
            train_loss = self.train_epoch(epoch)
            self.history['train_loss'].append(train_loss)
            
            # Validate
            val_loss, val_metrics = self.validate(epoch)
            self.history['val_loss'].append(val_loss)
            self.history['val_metrics'].append(val_metrics)
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Log to tensorboard
            self.writer.add_scalar('train/epoch_loss', train_loss, epoch)
            self.writer.add_scalar('val/loss', val_loss, epoch)
            
            for metric_name, metric_value in val_metrics.items():
                if isinstance(metric_value, (int, float)):
                    self.writer.add_scalar(f'val/{metric_name}', metric_value, epoch)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('train/learning_rate', current_lr, epoch)
            
            # Print epoch summary
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            
            for metric_name, metric_value in val_metrics.items():
                if isinstance(metric_value, (int, float)):
                    print(f"  Val {metric_name}: {metric_value:.4f}")
            
            print(f"  Learning Rate: {current_lr:.6f}")
            
            # Save best model
            if save_best and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.save_checkpoint(epoch, is_best=True)
                print(f"  *** New best model saved! ***")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Save periodic checkpoint
            if save_every is not None and epoch % save_every == 0:
                self.save_checkpoint(epoch, is_best=False)
            
            # Early stopping
            if early_stopping_patience is not None:
                if patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping triggered after {epoch} epochs")
                    print(f"Best epoch was {self.best_epoch} with val loss {self.best_val_loss:.4f}")
                    break
        
        # Save final model
        self.save_checkpoint(epoch, is_best=False, suffix='final')
        
        # Save training history
        self.save_history()
        
        print(f"\nTraining completed!")
        print(f"Best epoch: {self.best_epoch}")
        print(f"Best val loss: {self.best_val_loss:.4f}")
        
        return self.history
    
    def save_checkpoint(
        self,
        epoch: int,
        is_best: bool = False,
        suffix: str = '',
    ):
        """Save model checkpoint.
        
        Args:
            epoch: Current epoch
            is_best: Whether this is the best model
            suffix: Optional suffix for filename
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if is_best:
            path = self.checkpoint_dir / 'best_model.pt'
        elif suffix:
            path = self.checkpoint_dir / f'checkpoint_{suffix}.pt'
        else:
            path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        
        torch.save(checkpoint, path)
        print(f"  Checkpoint saved: {path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.history = checkpoint.get('history', self.history)
        
        print(f"Checkpoint loaded from {checkpoint_path}")
        print(f"Resuming from epoch {checkpoint['epoch']}")
    
    def save_history(self):
        """Save training history to JSON file."""
        history_path = self.checkpoint_dir / 'training_history.json'
        
        # Convert numpy values to Python types
        history_serializable = {}
        for key, values in self.history.items():
            if key == 'val_metrics':
                # Handle metrics dictionary
                serializable_metrics = []
                for metrics_dict in values:
                    serializable_dict = {}
                    for metric_name, metric_value in metrics_dict.items():
                        if isinstance(metric_value, (np.number, np.ndarray)):
                            serializable_dict[metric_name] = float(metric_value)
                        else:
                            serializable_dict[metric_name] = metric_value
                    serializable_metrics.append(serializable_dict)
                history_serializable[key] = serializable_metrics
            else:
                history_serializable[key] = [float(v) if isinstance(v, np.number) else v for v in values]
        
        with open(history_path, 'w') as f:
            json.dump(history_serializable, f, indent=2)
        
        print(f"Training history saved: {history_path}")


def create_optimizer(
    model: nn.Module,
    optimizer_name: str = 'adam',
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    **kwargs
) -> torch.optim.Optimizer:
    """Create optimizer from configuration.
    
    Args:
        model: Model to optimize
        optimizer_name: Name of optimizer ('adam', 'adamw', 'sgd')
        learning_rate: Learning rate
        weight_decay: Weight decay
        **kwargs: Additional optimizer parameters
        
    Returns:
        Optimizer instance
    """
    if optimizer_name.lower() == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs
        )
    elif optimizer_name.lower() == 'adamw':
        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs
        )
    elif optimizer_name.lower() == 'sgd':
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=kwargs.get('momentum', 0.9),
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str = 'cosine',
    n_epochs: int = 100,
    **kwargs
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Create learning rate scheduler from configuration.
    
    Args:
        optimizer: Optimizer to schedule
        scheduler_name: Name of scheduler ('cosine', 'step', 'plateau')
        n_epochs: Number of training epochs
        **kwargs: Additional scheduler parameters
        
    Returns:
        Scheduler instance or None
    """
    if scheduler_name is None or scheduler_name.lower() == 'none':
        return None
    
    if scheduler_name.lower() == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=n_epochs,
            **kwargs
        )
    elif scheduler_name.lower() == 'step':
        step_size = kwargs.get('step_size', 30)
        gamma = kwargs.get('gamma', 0.1)
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
        )
    elif scheduler_name.lower() == 'plateau':
        patience = kwargs.get('patience', 10)
        factor = kwargs.get('factor', 0.5)
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=patience,
            factor=factor,
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
