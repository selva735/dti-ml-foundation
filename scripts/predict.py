#!/usr/bin/env python
"""Prediction script for DTI prediction models.

This script makes predictions for new drug-target pairs using a trained model.
"""

import argparse
import os
import sys
from pathlib import Path
import torch
import pandas as pd
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data.molecular_graph import MolecularGraphGenerator
from data.protein_embedding import ProteinEmbeddingGenerator
from models.dti_model import create_dti_model
from models.uncertainty import UncertaintyEstimator
from utils.config import Config
from torch_geometric.data import Data, Batch


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Make DTI predictions')
    
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--input', type=str, help='Path to input CSV with drug-target pairs')
    parser.add_argument('--drug-smiles', type=str, help='Single drug SMILES')
    parser.add_argument('--protein-sequence', type=str, help='Single protein sequence')
    parser.add_argument('--output', type=str, default='predictions.csv', help='Output file path')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    parser.add_argument('--protein-model', type=str, default='facebook/esm2_t12_35M_UR50D', help='Protein model')
    parser.add_argument('--data-dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--uncertainty', action='store_true', help='Estimate uncertainty')
    parser.add_argument('--n-mc-samples', type=int, default=30, help='MC dropout samples')
    
    return parser.parse_args()


def predict_single(
    model,
    drug_smiles: str,
    protein_sequence: str,
    mol_generator: MolecularGraphGenerator,
    protein_generator: ProteinEmbeddingGenerator,
    device: str,
    uncertainty_estimator=None,
):
    """Make prediction for a single drug-target pair.
    
    Args:
        model: Trained model
        drug_smiles: Drug SMILES string
        protein_sequence: Protein sequence
        mol_generator: Molecular graph generator
        protein_generator: Protein embedding generator
        device: Device to run on
        uncertainty_estimator: Uncertainty estimator (optional)
        
    Returns:
        Dictionary with prediction and uncertainty (if requested)
    """
    # Generate molecular graph
    drug_graph = mol_generator.smiles_to_graph(drug_smiles)
    if drug_graph is None:
        print(f"Warning: Invalid SMILES: {drug_smiles}")
        return None
    
    # Get protein embedding
    protein_emb = protein_generator.get_embedding(protein_sequence, pooling='mean')
    
    # Convert to tensors
    drug_batch = Batch.from_data_list([drug_graph]).to(device)
    protein_tensor = torch.FloatTensor(protein_emb).unsqueeze(0).to(device)
    
    # Make prediction
    model.eval()
    
    if uncertainty_estimator is not None:
        # Predict with uncertainty
        result = uncertainty_estimator.predict(drug_batch, protein_tensor)
        
        prediction = result['mean'].cpu().item()
        uncertainty = result['std'].cpu().item()
        
        return {
            'prediction': prediction,
            'uncertainty': uncertainty,
        }
    else:
        # Standard prediction
        with torch.no_grad():
            pred, _ = model(drug_batch, protein_tensor)
        
        prediction = pred.cpu().item()
        
        return {
            'prediction': prediction,
        }


def main():
    """Main prediction function."""
    args = parse_args()
    
    print("=" * 70)
    print("DTI Prediction")
    print("=" * 70)
    
    # Load checkpoint
    print(f"\nLoading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    
    # Initialize generators
    print("\nInitializing generators...")
    mol_generator = MolecularGraphGenerator()
    protein_generator = ProteinEmbeddingGenerator(
        model_name=args.protein_model,
        cache_dir=os.path.join(args.data_dir, 'embeddings_cache'),
        device=args.device,
    )
    
    # Get feature dimensions
    node_feat_dim = mol_generator.get_node_feature_dim()
    edge_feat_dim = mol_generator.get_edge_feature_dim()
    protein_emb_dim = protein_generator.embedding_dim
    
    feature_dims = {
        'node_feat_dim': node_feat_dim,
        'edge_feat_dim': edge_feat_dim,
        'protein_emb_dim': protein_emb_dim,
    }
    
    # Load config
    if args.config:
        config = Config(args.config)
        model_config = config['model']
    else:
        model_config = {
            'hidden_dim': 256,
            'gnn_layers': 3,
            'gnn_type': 'gat',
            'dropout': 0.2,
        }
    
    # Create model
    print("\nCreating model...")
    model = create_dti_model(feature_dims, model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(args.device)
    model.eval()
    
    # Set up uncertainty estimator if needed
    uncertainty_estimator = None
    if args.uncertainty:
        print("\nSetting up uncertainty estimation...")
        uncertainty_estimator = UncertaintyEstimator(
            model,
            method='mcdropout',
            n_samples=args.n_mc_samples,
        )
    
    # Single prediction mode
    if args.drug_smiles and args.protein_sequence:
        print("\nMaking single prediction...")
        print(f"Drug SMILES: {args.drug_smiles}")
        print(f"Protein sequence length: {len(args.protein_sequence)}")
        
        result = predict_single(
            model,
            args.drug_smiles,
            args.protein_sequence,
            mol_generator,
            protein_generator,
            args.device,
            uncertainty_estimator,
        )
        
        if result is not None:
            print("\nPrediction Results:")
            print(f"  Affinity: {result['prediction']:.4f}")
            
            if 'uncertainty' in result:
                print(f"  Uncertainty: {result['uncertainty']:.4f}")
                
                # Compute confidence interval
                lower = result['prediction'] - 1.96 * result['uncertainty']
                upper = result['prediction'] + 1.96 * result['uncertainty']
                print(f"  95% CI: [{lower:.4f}, {upper:.4f}]")
        else:
            print("Prediction failed!")
    
    # Batch prediction mode
    elif args.input:
        print(f"\nLoading input from {args.input}...")
        input_df = pd.read_csv(args.input)
        
        print(f"Making predictions for {len(input_df)} drug-target pairs...")
        
        predictions = []
        uncertainties = []
        
        from tqdm import tqdm
        for idx, row in tqdm(input_df.iterrows(), total=len(input_df)):
            result = predict_single(
                model,
                row['smiles'],
                row['sequence'],
                mol_generator,
                protein_generator,
                args.device,
                uncertainty_estimator,
            )
            
            if result is not None:
                predictions.append(result['prediction'])
                if 'uncertainty' in result:
                    uncertainties.append(result['uncertainty'])
            else:
                predictions.append(None)
                uncertainties.append(None)
        
        # Add predictions to dataframe
        input_df['prediction'] = predictions
        
        if args.uncertainty:
            input_df['uncertainty'] = uncertainties
        
        # Save results
        input_df.to_csv(args.output, index=False)
        print(f"\nPredictions saved to {args.output}")
        
        # Print summary statistics
        import numpy as np
        valid_preds = [p for p in predictions if p is not None]
        
        if valid_preds:
            print("\nPrediction Summary:")
            print(f"  Total predictions: {len(valid_preds)}")
            print(f"  Mean affinity: {np.mean(valid_preds):.4f}")
            print(f"  Std affinity: {np.std(valid_preds):.4f}")
            print(f"  Min affinity: {np.min(valid_preds):.4f}")
            print(f"  Max affinity: {np.max(valid_preds):.4f}")
    
    else:
        print("\nError: Must provide either --input or both --drug-smiles and --protein-sequence")
        return
    
    print("\nPrediction complete!")


if __name__ == '__main__':
    main()
