"""
Evaluation script for multi-modal DTI prediction with cold-start scenarios.
"""

import os
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
from tqdm import tqdm

from train import DTIModel
from data import DTIDataset, ColdStartSplit


def concordance_index(y_true, y_pred):
    """
    Compute concordance index (C-index) for ranking evaluation.
    
    Args:
        y_true: True affinity values
        y_pred: Predicted affinity values
        
    Returns:
        C-index score
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    n = len(y_true)
    concordant = 0
    total_pairs = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            if y_true[i] != y_true[j]:
                total_pairs += 1
                if (y_true[i] > y_true[j] and y_pred[i] > y_pred[j]) or \
                   (y_true[i] < y_true[j] and y_pred[i] < y_pred[j]):
                    concordant += 1
                elif y_pred[i] == y_pred[j]:
                    concordant += 0.5
    
    if total_pairs == 0:
        return 0.0
    
    return concordant / total_pairs


def compute_metrics(y_true, y_pred, metrics=['mse', 'mae', 'rmse', 'r2', 'pearson', 'spearman', 'ci']):
    """
    Compute evaluation metrics.
    
    Args:
        y_true: True affinity values
        y_pred: Predicted affinity values
        metrics: List of metrics to compute
        
    Returns:
        Dictionary of computed metrics
    """
    results = {}
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if 'mse' in metrics:
        results['mse'] = mean_squared_error(y_true, y_pred)
    
    if 'mae' in metrics:
        results['mae'] = mean_absolute_error(y_true, y_pred)
    
    if 'rmse' in metrics:
        results['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
    
    if 'r2' in metrics:
        results['r2'] = r2_score(y_true, y_pred)
    
    if 'pearson' in metrics:
        corr, pval = pearsonr(y_true, y_pred)
        results['pearson'] = corr
        results['pearson_pval'] = pval
    
    if 'spearman' in metrics:
        corr, pval = spearmanr(y_true, y_pred)
        results['spearman'] = corr
        results['spearman_pval'] = pval
    
    if 'ci' in metrics:
        results['ci'] = concordance_index(y_true, y_pred)
    
    return results


def evaluate_model(model, dataloader, device, config):
    """
    Evaluate model on a dataset.
    
    Args:
        model: DTI prediction model
        dataloader: Data loader
        device: Device to run on
        config: Configuration dictionary
        
    Returns:
        Predictions, uncertainties, and true values
    """
    model.eval()
    
    all_predictions = []
    all_uncertainties = []
    all_true_values = []
    all_epistemic = []
    all_aleatoric = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Placeholder for actual evaluation
            # In practice, process batch and make predictions
            
            # output, attention_weights = model(drug_data, protein_data, tda_data, return_uncertainty=True)
            
            # Collect predictions and uncertainties
            pass
    
    return {
        'predictions': all_predictions,
        'uncertainties': all_uncertainties,
        'true_values': all_true_values,
        'epistemic': all_epistemic,
        'aleatoric': all_aleatoric
    }


def evaluate_cold_start(model, dataset, split_info, device, config, split_type):
    """
    Evaluate model on cold-start scenarios.
    
    Args:
        model: Trained model
        dataset: DTI dataset
        split_info: Split information
        device: Device to run on
        config: Configuration
        split_type: Type of cold-start split
        
    Returns:
        Evaluation results
    """
    print(f"\nEvaluating {split_type} scenario...")
    
    # Create test subset
    test_indices = split_info['test']
    
    # Placeholder for evaluation
    # test_dataset = Subset(dataset, test_indices)
    # test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'])
    
    # results = evaluate_model(model, test_loader, device, config)
    
    # Placeholder results
    results = {
        'predictions': [],
        'true_values': [],
        'uncertainties': []
    }
    
    # Compute metrics
    if len(results['predictions']) > 0:
        metrics = compute_metrics(
            results['true_values'],
            results['predictions'],
            metrics=config['evaluation']['metrics']
        )
    else:
        metrics = {metric: 0.0 for metric in config['evaluation']['metrics']}
    
    return {
        'metrics': metrics,
        'predictions': results
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate DTI prediction model')
    parser.add_argument('--config', type=str, default='config/default_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='davis',
                       choices=['davis', 'kiba', 'bindingdb'],
                       help='Dataset to use')
    parser.add_argument('--split_type', type=str, default='random',
                       choices=['random', 'cold_drug', 'cold_target', 'cold_both'],
                       help='Type of data split')
    parser.add_argument('--split_info', type=str,
                       help='Path to split info JSON file')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    parser.add_argument('--output_dir', type=str, default='./eval_results',
                       help='Output directory for results')
    parser.add_argument('--evaluate_all_splits', action='store_true',
                       help='Evaluate on all cold-start splits')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config
    config['data']['dataset'] = args.dataset
    config['data']['split_type'] = args.split_type
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print("Loading model...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = DTIModel(checkpoint['config']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Load dataset
    print(f"Loading {args.dataset} dataset...")
    dataset = DTIDataset(
        root=config['data']['data_root'],
        dataset_name=args.dataset
    )
    
    # Load or generate split info
    if args.split_info and os.path.exists(args.split_info):
        with open(args.split_info, 'r') as f:
            split_info = json.load(f)
        print(f"Loaded split info from {args.split_info}")
    else:
        print("Generating split...")
        splitter = ColdStartSplit(
            dataset,
            split_type=args.split_type,
            test_ratio=config['data']['test_ratio'],
            val_ratio=config['data']['val_ratio'],
            random_seed=config['data']['random_seed']
        )
        split_info = splitter.generate_split()
    
    # Evaluate
    if args.evaluate_all_splits:
        # Evaluate on all cold-start scenarios
        split_types = ['random', 'cold_drug', 'cold_target', 'cold_both']
        all_results = {}
        
        for split_type in split_types:
            print(f"\n{'='*50}")
            print(f"Evaluating {split_type.upper()} scenario")
            print('='*50)
            
            # Generate split for this scenario
            splitter = ColdStartSplit(
                dataset,
                split_type=split_type,
                test_ratio=config['data']['test_ratio'],
                val_ratio=config['data']['val_ratio'],
                random_seed=config['data']['random_seed']
            )
            split_info = splitter.generate_split()
            
            # Evaluate
            results = evaluate_cold_start(
                model, dataset, split_info, device, config, split_type
            )
            all_results[split_type] = results
            
            # Print metrics
            print(f"\n{split_type.upper()} Results:")
            for metric, value in results['metrics'].items():
                print(f"  {metric}: {value:.4f}")
        
        # Save all results
        output_file = os.path.join(args.output_dir, f'all_splits_{args.dataset}.json')
        with open(output_file, 'w') as f:
            # Convert numpy types to python types for JSON serialization
            json_results = {}
            for split_type, results in all_results.items():
                json_results[split_type] = {
                    'metrics': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                              for k, v in results['metrics'].items()}
                }
            json.dump(json_results, f, indent=2)
        
        print(f"\nSaved results to {output_file}")
        
        # Create comparison table
        print("\n" + "="*80)
        print("COLD-START EVALUATION SUMMARY")
        print("="*80)
        
        metrics_to_show = ['mse', 'rmse', 'mae', 'r2', 'pearson', 'ci']
        print(f"{'Metric':<15} " + " ".join([f"{st:>12}" for st in split_types]))
        print("-"*80)
        
        for metric in metrics_to_show:
            if metric in all_results['random']['metrics']:
                values = [all_results[st]['metrics'].get(metric, 0.0) for st in split_types]
                print(f"{metric:<15} " + " ".join([f"{v:>12.4f}" for v in values]))
        
    else:
        # Evaluate on single split
        results = evaluate_cold_start(
            model, dataset, split_info, device, config, args.split_type
        )
        
        # Print metrics
        print(f"\n{'='*50}")
        print(f"{args.split_type.upper()} Evaluation Results")
        print('='*50)
        
        for metric, value in results['metrics'].items():
            print(f"{metric}: {value:.4f}")
        
        # Save results
        output_file = os.path.join(
            args.output_dir,
            f'{args.dataset}_{args.split_type}_results.json'
        )
        
        with open(output_file, 'w') as f:
            json_results = {
                'metrics': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                          for k, v in results['metrics'].items()}
            }
            json.dump(json_results, f, indent=2)
        
        print(f"\nSaved results to {output_file}")
        
        # Save predictions if requested
        if config['evaluation']['save_predictions'] and len(results['predictions']['predictions']) > 0:
            predictions_df = pd.DataFrame({
                'true_affinity': results['predictions']['true_values'],
                'predicted_affinity': results['predictions']['predictions'],
                'uncertainty': results['predictions']['uncertainties']
            })
            
            pred_file = os.path.join(
                args.output_dir,
                f'{args.dataset}_{args.split_type}_predictions.csv'
            )
            predictions_df.to_csv(pred_file, index=False)
            print(f"Saved predictions to {pred_file}")


if __name__ == '__main__':
    main()
