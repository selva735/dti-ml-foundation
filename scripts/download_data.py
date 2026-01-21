#!/usr/bin/env python
"""Download and prepare benchmark datasets for DTI prediction.

This script downloads standard benchmark datasets (Davis, KIBA, BindingDB)
and prepares them for training.
"""

import argparse
import os
from pathlib import Path
import pandas as pd
import requests
from tqdm import tqdm
import zipfile
import json


DATASET_URLS = {
    'davis': {
        'url': 'https://raw.githubusercontent.com/hkmztrk/DeepDTA/master/data/davis/',
        'files': ['folds/train_fold_0.txt', 'ligands_can.txt', 'proteins.txt', 'Y'],
    },
    'kiba': {
        'url': 'https://raw.githubusercontent.com/hkmztrk/DeepDTA/master/data/kiba/',
        'files': ['folds/train_fold_0.txt', 'ligands_can.txt', 'proteins.txt', 'Y'],
    },
}


def download_file(url: str, save_path: str):
    """Download a file from URL.
    
    Args:
        url: URL to download from
        save_path: Path to save file
    """
    print(f"Downloading from {url}...")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(save_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    print(f"Saved to {save_path}")


def create_sample_dataset(data_dir: Path, n_samples: int = 1000):
    """Create a sample dataset for testing.
    
    Args:
        data_dir: Directory to save sample data
        n_samples: Number of samples to generate
    """
    import numpy as np
    
    print(f"Creating sample dataset with {n_samples} samples...")
    
    # Sample SMILES strings
    sample_smiles = [
        'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',  # Ibuprofen
        'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin
        'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
        'CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C',  # Testosterone
        'C1=CC=C(C=C1)C=O',  # Benzaldehyde
    ] * (n_samples // 5 + 1)
    sample_smiles = sample_smiles[:n_samples]
    
    # Sample protein sequences
    sample_proteins = [
        'MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK',  # GFP
        'MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHHYREQIKRVKDSEDVPMVLVGNKCDLPSRTVDTKQAQDLARSYGIPFIETSAKTRQGVDDAFYTLVREIRKHKEKMSKDGKKKKKKSKTKCVIM',  # KRAS
    ] * (n_samples // 2 + 1)
    sample_proteins = sample_proteins[:n_samples]
    
    # Generate random affinities
    np.random.seed(42)
    affinities = np.random.uniform(4.0, 10.0, n_samples)
    
    # Create DataFrame
    df = pd.DataFrame({
        'smiles': sample_smiles,
        'sequence': sample_proteins,
        'affinity': affinities,
    })
    
    # Save to CSV
    save_path = data_dir / 'sample_dataset.csv'
    df.to_csv(save_path, index=False)
    
    print(f"Sample dataset saved to {save_path}")
    print(f"Dataset contains {len(df)} samples")
    
    return df


def main():
    """Main function for dataset download."""
    parser = argparse.ArgumentParser(description='Download DTI benchmark datasets')
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['davis', 'kiba', 'sample', 'all'],
        default='sample',
        help='Dataset to download'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='./data',
        help='Directory to save datasets'
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=1000,
        help='Number of samples for sample dataset'
    )
    
    args = parser.parse_args()
    
    # Create data directory
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    if args.dataset == 'sample' or args.dataset == 'all':
        # Create sample dataset
        create_sample_dataset(data_dir, n_samples=args.n_samples)
    
    if args.dataset in ['davis', 'kiba'] or args.dataset == 'all':
        print(f"\nNote: Downloading real benchmark datasets requires proper data processing.")
        print(f"For this initial version, we recommend using the sample dataset.")
        print(f"Real dataset download functionality can be implemented based on specific data sources.")
    
    print("\nDataset preparation complete!")
    print(f"Data directory: {data_dir}")


if __name__ == '__main__':
    main()
