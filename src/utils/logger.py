"""Logging utilities for DTI prediction.

This module provides logging configuration and utilities for consistent logging
across the project.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logger(
    name: str = 'dti-ml-foundation',
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """Set up logger with console and optional file handlers.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Path to log file (optional)
        format_string: Custom format string (optional)
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Default format
    if format_string is None:
        format_string = '[%(asctime)s] %(levelname)s - %(name)s - %(message)s'
    
    formatter = logging.Formatter(format_string, datefmt='%Y-%m-%d %H:%M:%S')
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = 'dti-ml-foundation') -> logging.Logger:
    """Get existing logger or create new one.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        # Set up default logger if not configured
        logger = setup_logger(name)
    
    return logger


class TrainingLogger:
    """Logger for training progress and metrics.
    
    Provides structured logging for training experiments.
    """
    
    def __init__(
        self,
        experiment_name: str,
        log_dir: str = './logs',
        level: int = logging.INFO,
    ):
        """Initialize training logger.
        
        Args:
            experiment_name: Name of the experiment
            log_dir: Directory to save logs
            level: Logging level
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.log_dir / f'{experiment_name}_{timestamp}.log'
        
        # Set up logger
        self.logger = setup_logger(
            name=f'training.{experiment_name}',
            level=level,
            log_file=str(log_file),
        )
    
    def log_config(self, config: dict):
        """Log configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.logger.info("="*60)
        self.logger.info(f"Experiment: {self.experiment_name}")
        self.logger.info("="*60)
        self.logger.info("Configuration:")
        
        self._log_dict(config, indent=2)
    
    def _log_dict(self, d: dict, indent: int = 0):
        """Recursively log dictionary.
        
        Args:
            d: Dictionary to log
            indent: Indentation level
        """
        prefix = " " * indent
        
        for key, value in d.items():
            if isinstance(value, dict):
                self.logger.info(f"{prefix}{key}:")
                self._log_dict(value, indent + 2)
            else:
                self.logger.info(f"{prefix}{key}: {value}")
    
    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        metrics: dict,
        learning_rate: float,
    ):
        """Log epoch results.
        
        Args:
            epoch: Epoch number
            train_loss: Training loss
            val_loss: Validation loss
            metrics: Dictionary of metrics
            learning_rate: Current learning rate
        """
        self.logger.info("="*60)
        self.logger.info(f"Epoch {epoch}")
        self.logger.info("-"*60)
        self.logger.info(f"Train Loss: {train_loss:.6f}")
        self.logger.info(f"Val Loss:   {val_loss:.6f}")
        
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                self.logger.info(f"{metric_name}: {metric_value:.6f}")
        
        self.logger.info(f"Learning Rate: {learning_rate:.8f}")
        self.logger.info("="*60)
    
    def log_best_model(self, epoch: int, val_loss: float):
        """Log best model information.
        
        Args:
            epoch: Epoch number
            val_loss: Validation loss
        """
        self.logger.info(f"*** New best model at epoch {epoch} with val loss {val_loss:.6f} ***")
    
    def log_early_stopping(self, epoch: int, patience: int):
        """Log early stopping trigger.
        
        Args:
            epoch: Epoch where training stopped
            patience: Patience value
        """
        self.logger.info(f"Early stopping triggered at epoch {epoch} (patience: {patience})")
    
    def log_completion(self, best_epoch: int, best_val_loss: float):
        """Log training completion.
        
        Args:
            best_epoch: Best epoch number
            best_val_loss: Best validation loss
        """
        self.logger.info("="*60)
        self.logger.info("Training Completed")
        self.logger.info("-"*60)
        self.logger.info(f"Best Epoch: {best_epoch}")
        self.logger.info(f"Best Val Loss: {best_val_loss:.6f}")
        self.logger.info("="*60)
    
    def info(self, message: str):
        """Log info message.
        
        Args:
            message: Message to log
        """
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message.
        
        Args:
            message: Message to log
        """
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message.
        
        Args:
            message: Message to log
        """
        self.logger.error(message)
