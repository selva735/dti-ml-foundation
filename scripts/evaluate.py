#!/usr/bin/env python
"""Evaluation script for DTI prediction models.

This script evaluates trained models on test sets and cold-start scenarios.
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

from data.dataset import DTIDataset, create_dataloader
from data.preprocessing import DTIPreprocessor
from models.dti_model import create_dti_model
from training.evaluator import DTIEvaluator, print_metrics
from training.cold_start import ColdStartEvaluator, print_cold_start_results
from models.uncertainty import UncertaintyEstimator
from utils.config import Config
import utils.visualization as viz


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate DTI prediction model')
    
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data-path', type=str, required=True, help='Path to test dataset CSV')
    parser.add_argument('--train-data-path', type=str, help='Path to training dataset (for cold-start)')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--output-dir', type=str, default='./eval_results', help='Output directory')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    parser.add_argument('--protein-model', type=str, default='facebook/esm2_t12_35M_UR50D', help='Protein model')
    parser.add_argument('--data-dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--cold-start', action='store_true', help='Perform cold-start evaluation')
    parser.add_argument('--uncertainty', action='store_true', help='Estimate uncertainty')
    parser.add_argument('--n-mc-samples', type=int, default=30, help='MC dropout samples for uncertainty')
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("DTI Model Evaluation")
    print("=" * 70)
    
    # Load checkpoint
    print(f"\nLoading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    
    # Load test data
    print(f"\nLoading test data from {args.data_path}...")
    test_df = pd.read_csv(args.data_path)
    print(f"Loaded {len(test_df)} test samples")
    
    # Preprocess
    preprocessor = DTIPreprocessor()
    test_df = preprocessor.clean_data(test_df)
    
    # Create test dataset
    print("\nCreating test dataset...")
    test_dataset = DTIDataset(
        test_df,
        protein_model=args.protein_model,
        cache_dir=os.path.join(args.data_dir, 'embeddings_cache'),
    )
    
    test_loader = create_dataloader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    
    # Get feature dimensions
    feature_dims = test_dataset.get_feature_dims()
    
    # Load config if provided
    if args.config:
        config = Config(args.config)
        model_config = config['model']
    else:
        # Try to extract from checkpoint or use defaults
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
    
    # Create evaluator
    evaluator = DTIEvaluator(task_type='regression')
    
    # Standard evaluation
    print("\nPerforming standard evaluation...")
    results = evaluator.evaluate_model(
        model,
        test_loader,
        device=args.device,
        return_predictions=True,
    )
    
    predictions = results.pop('predictions')
    targets = results.pop('targets')
    
    # Print metrics
    print_metrics(results, prefix="Test Set ")
    
    # Save metrics
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump({k: float(v) for k, v in results.items()}, f, indent=2)
    
    # Plot predictions
    print("\nGenerating prediction plots...")
    viz.plot_predictions(
        targets,
        predictions,
        title='Test Set Predictions',
        save_path=str(output_dir / 'predictions.png'),
    )
    
    viz.plot_residuals(
        targets,
        predictions,
        save_path=str(output_dir / 'residuals.png'),
    )
    
    # Cold-start evaluation
    if args.cold_start and args.train_data_path:
        print("\nPerforming cold-start evaluation...")
        
        # Load training data
        train_df = pd.read_csv(args.train_data_path)
        train_drugs = set(train_df['smiles'].unique())
        train_targets = set(train_df['sequence'].unique())
        
        # Identify cold-start samples
        cold_start_eval = ColdStartEvaluator()
        scenario_masks = cold_start_eval.identify_cold_start_samples(
            test_df['smiles'].tolist(),
            test_df['sequence'].tolist(),
            train_drugs,
            train_targets,
        )
        
        # Evaluate by scenario
        cold_start_results = cold_start_eval.evaluate_by_scenario(
            predictions,
            targets,
            scenario_masks,
            evaluator,
        )
        
        print_cold_start_results(cold_start_results)
        
        # Save results
        cold_start_serializable = {}
        for scenario, metrics in cold_start_results.items():
            cold_start_serializable[scenario] = {
                k: float(v) for k, v in metrics.items()
            }
        
        with open(output_dir / 'cold_start_metrics.json', 'w') as f:
            json.dump(cold_start_serializable, f, indent=2)
        
        # Plot comparison
        viz.plot_cold_start_comparison(
            cold_start_results,
            metric='rmse',
            save_path=str(output_dir / 'cold_start_comparison.png'),
        )
    
    # Uncertainty estimation
    if args.uncertainty:
        print("\nEstimating prediction uncertainty...")
        
        uncertainty_estimator = UncertaintyEstimator(
            model,
            method='mcdropout',
            n_samples=args.n_mc_samples,
        )
        
        # Get predictions with uncertainty
        all_uncertainties = []
        
        for batch in test_loader:
            drug_graph = batch['drug_graph'].to(args.device)
            protein_emb = batch['protein_emb'].to(args.device)
            
            result = uncertainty_estimator.predict(drug_graph, protein_emb)
            all_uncertainties.append(result['std'].cpu().numpy())
        
        import numpy as np
        uncertainties = np.concatenate(all_uncertainties, axis=0)
        
        # Plot uncertainty
        viz.plot_uncertainty(
            predictions,
            uncertainties,
            targets=targets,
            save_path=str(output_dir / 'uncertainty.png'),
        )
        
        print(f"Mean uncertainty: {uncertainties.mean():.4f}")
        print(f"Std uncertainty: {uncertainties.std():.4f}")
    
    # Save predictions
    print("\nSaving predictions...")
    results_df = test_df.copy()
    results_df['prediction'] = predictions.flatten()
    results_df['target'] = targets.flatten()
    
    if args.uncertainty:
        results_df['uncertainty'] = uncertainties.flatten()
    
    results_df.to_csv(output_dir / 'predictions.csv', index=False)
    
    print(f"\nEvaluation complete! Results saved to {output_dir}")


if __name__ == '__main__':
    main()
