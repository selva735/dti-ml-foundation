#!/usr/bin/env python
"""Main training script for DTI prediction models.

This script provides a command-line interface for training DTI prediction models
with various configurations.
"""

import argparse
import os
import sys
from pathlib import Path
import torch
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data.dataset import DTIDataset, create_dataloader
from data.preprocessing import DTIPreprocessor
from models.dti_model import create_dti_model
from training.trainer import Trainer, create_optimizer, create_scheduler
from training.evaluator import DTIEvaluator
from utils.config import Config, create_default_config
from utils.logger import TrainingLogger
import utils.visualization as viz


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Train DTI prediction model')
    
    # Data arguments
    parser.add_argument('--data-path', type=str, required=True, help='Path to dataset CSV file')
    parser.add_argument('--data-dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set fraction')
    parser.add_argument('--val-size', type=float, default=0.1, help='Validation set fraction')
    
    # Model arguments
    parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--gnn-layers', type=int, default=3, help='Number of GNN layers')
    parser.add_argument('--gnn-type', type=str, default='gat', choices=['gat', 'gin'], help='GNN type')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--protein-model', type=str, default='facebook/esm2_t12_35M_UR50D', help='Protein language model')
    
    # Training arguments
    parser.add_argument('--n-epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw', 'sgd'], help='Optimizer')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'step', 'plateau', 'none'], help='LR scheduler')
    parser.add_argument('--early-stopping', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--gradient-clip', type=float, default=1.0, help='Gradient clipping value')
    
    # Other arguments
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--log-dir', type=str, default='./runs', help='TensorBoard log directory')
    parser.add_argument('--config', type=str, help='Path to config YAML file')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num-workers', type=int, default=0, help='Number of data loading workers')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Set up logger
    logger = TrainingLogger('dti-training', log_dir=args.log_dir)
    
    logger.info("=" * 70)
    logger.info("DTI-ML-Foundation Training")
    logger.info("=" * 70)
    
    # Load configuration
    if args.config:
        config = Config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    else:
        config = Config()
        config.config = create_default_config()
        logger.info("Using default configuration")
    
    # Override config with command-line arguments
    config['data']['batch_size'] = args.batch_size
    config['training']['n_epochs'] = args.n_epochs
    config['training']['learning_rate'] = args.learning_rate
    config['model']['hidden_dim'] = args.hidden_dim
    config['model']['gnn_layers'] = args.gnn_layers
    
    logger.log_config(config.to_dict())
    
    # Load data
    logger.info("\nLoading dataset...")
    df = pd.read_csv(args.data_path)
    logger.info(f"Loaded {len(df)} samples from {args.data_path}")
    
    # Preprocess data
    logger.info("\nPreprocessing data...")
    preprocessor = DTIPreprocessor()
    df_clean = preprocessor.clean_data(df)
    
    # Split data
    train_df, val_df, test_df = preprocessor.split_data(
        df_clean,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.seed,
    )
    
    logger.info(f"Train samples: {len(train_df)}")
    logger.info(f"Val samples: {len(val_df)}")
    logger.info(f"Test samples: {len(test_df)}")
    
    # Normalize targets
    train_df = preprocessor.normalize_targets(train_df, method='standardize', fit=True)
    val_df = preprocessor.normalize_targets(val_df, method='standardize', fit=False)
    test_df = preprocessor.normalize_targets(test_df, method='standardize', fit=False)
    
    # Create datasets
    logger.info("\nCreating datasets...")
    train_dataset = DTIDataset(
        train_df,
        protein_model=args.protein_model,
        cache_dir=os.path.join(args.data_dir, 'embeddings_cache'),
    )
    
    val_dataset = DTIDataset(
        val_df,
        protein_model=args.protein_model,
        cache_dir=os.path.join(args.data_dir, 'embeddings_cache'),
    )
    
    # Get feature dimensions
    feature_dims = train_dataset.get_feature_dims()
    logger.info(f"Feature dimensions: {feature_dims}")
    
    # Create data loaders
    train_loader = create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    
    val_loader = create_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    
    # Create model
    logger.info("\nCreating model...")
    model = create_dti_model(feature_dims, config['model'])
    
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {n_params:,}")
    logger.info(f"Trainable parameters: {n_trainable:,}")
    
    # Create optimizer
    optimizer = create_optimizer(
        model,
        optimizer_name=args.optimizer,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    
    # Create scheduler
    scheduler = create_scheduler(
        optimizer,
        scheduler_name=args.scheduler,
        n_epochs=args.n_epochs,
    )
    
    # Create evaluator
    evaluator = DTIEvaluator(task_type='regression')
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=torch.nn.MSELoss(),
        device=args.device,
        scheduler=scheduler,
        evaluator=evaluator,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        gradient_clip=args.gradient_clip,
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train
    logger.info("\nStarting training...")
    history = trainer.train(
        n_epochs=args.n_epochs,
        early_stopping_patience=args.early_stopping,
        save_best=True,
        save_every=10,
    )
    
    # Plot training history
    logger.info("\nGenerating training plots...")
    viz.plot_training_history(
        history,
        save_path=os.path.join(args.checkpoint_dir, 'training_history.png'),
    )
    
    logger.info("\nTraining completed successfully!")
    logger.info(f"Best model saved to: {os.path.join(args.checkpoint_dir, 'best_model.pt')}")


if __name__ == '__main__':
    main()
