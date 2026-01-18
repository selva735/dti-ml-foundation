"""
Training script for multi-modal DTI prediction.
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
import json

from models import DrugGNN, ProteinPLM, TDAEncoder, FusionAttention, EvidentialHead
from models.evidential_head import evidential_loss, evidential_mse_loss
from data import DTIDataset, ColdStartSplit
from data.preprocessing import DrugPreprocessor, ProteinPreprocessor, TDAPreprocessor


class DTIModel(nn.Module):
    """
    Complete multi-modal DTI prediction model.
    """
    
    def __init__(self, config):
        super(DTIModel, self).__init__()
        
        # Initialize encoders
        self.drug_gnn = DrugGNN(**config['model']['drug_gnn'])
        self.protein_plm = ProteinPLM(**config['model']['protein_plm'])
        self.tda_encoder = TDAEncoder(**config['model']['tda_encoder'])
        
        # Fusion module
        self.fusion = FusionAttention(**config['model']['fusion_attention'])
        
        # Prediction head
        self.prediction_head = EvidentialHead(**config['model']['evidential_head'])
        
    def forward(self, drug_data, protein_data, tda_data, return_uncertainty=False):
        """
        Forward pass through the complete model.
        
        Args:
            drug_data: Drug molecular graph data
            protein_data: Protein sequence data
            tda_data: Topological features
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            Predictions and optionally uncertainty estimates
        """
        # Encode modalities
        drug_embed = self.drug_gnn(
            drug_data['node_features'],
            drug_data['edge_index'],
            drug_data['batch']
        )
        
        protein_embed = self.protein_plm(
            protein_data['tokens'],
            mask=protein_data.get('mask')
        )
        
        tda_embed = self.tda_encoder(
            tda_data['persistence_features'],
            tda_data['betti_curves'],
            tda_data['landscapes']
        )
        
        # Fuse modalities
        fused_embed, attention_weights = self.fusion(
            drug_embed, protein_embed, tda_embed
        )
        
        # Predict affinity
        if return_uncertainty:
            output = self.prediction_head(fused_embed, return_all=True)
            return output, attention_weights
        else:
            pred = self.prediction_head(fused_embed, return_all=False)
            return pred


def train_epoch(model, dataloader, optimizer, device, config):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        # Move data to device
        # Note: Simplified data loading - in practice, use proper batching
        
        optimizer.zero_grad()
        
        # Placeholder for actual training loop
        # In practice, extract drug_data, protein_data, tda_data from batch
        
        # Forward pass
        # output = model(drug_data, protein_data, tda_data, return_uncertainty=True)
        
        # Compute loss
        # loss = evidential_loss(...)
        
        # For demonstration:
        loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if config['training'].get('gradient_clip'):
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config['training']['gradient_clip']
            )
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / max(num_batches, 1)


def validate(model, dataloader, device, config):
    """Validate the model."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            # Placeholder for validation loop
            loss = torch.tensor(0.0, device=device)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / max(num_batches, 1)


def main():
    parser = argparse.ArgumentParser(description='Train DTI prediction model')
    parser.add_argument('--config', type=str, default='config/default_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--dataset', type=str, default='davis',
                       choices=['davis', 'kiba', 'bindingdb'],
                       help='Dataset to use')
    parser.add_argument('--split_type', type=str, default='random',
                       choices=['random', 'cold_drug', 'cold_target', 'cold_both'],
                       help='Type of data split')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use for training')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory for checkpoints and logs')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments
    config['data']['dataset'] = args.dataset
    config['data']['split_type'] = args.split_type
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(config['logging']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['logging']['log_dir'], exist_ok=True)
    
    # Save configuration
    with open(os.path.join(args.output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    # Initialize dataset
    print(f"Loading {args.dataset} dataset...")
    dataset = DTIDataset(
        root=config['data']['data_root'],
        dataset_name=args.dataset
    )
    
    # Create splits
    print(f"Creating {args.split_type} split...")
    splitter = ColdStartSplit(
        dataset,
        split_type=args.split_type,
        test_ratio=config['data']['test_ratio'],
        val_ratio=config['data']['val_ratio'],
        random_seed=config['data']['random_seed']
    )
    split_info = splitter.generate_split()
    
    # Save split info
    with open(os.path.join(args.output_dir, 'split_info.json'), 'w') as f:
        json.dump(split_info, f, indent=2)
    
    # Create data loaders
    # Note: In practice, implement proper data loading with collate functions
    # train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'])
    
    # Initialize model
    print("Initializing model...")
    model = DTIModel(config).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")
    
    # Initialize optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Initialize scheduler
    if config['training']['scheduler'] == 'reduce_on_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=config['training']['scheduler_patience'],
            factor=config['training']['scheduler_factor']
        )
    
    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config['training']['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['training']['num_epochs']}")
        
        # Train
        # train_loss = train_epoch(model, train_loader, optimizer, device, config)
        train_loss = 0.0  # Placeholder
        
        # Validate
        # val_loss = validate(model, val_loader, device, config)
        val_loss = 0.0  # Placeholder
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Learning rate scheduling
        if config['training']['scheduler'] == 'reduce_on_plateau':
            scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            checkpoint_path = os.path.join(
                config['logging']['checkpoint_dir'],
                f'best_model_{args.dataset}_{args.split_type}.pt'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config['training']['early_stopping_patience']:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break
    
    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()
