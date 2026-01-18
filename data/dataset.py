"""
DTI Dataset classes for Davis, KIBA, and BindingDB.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class DTIDataset(Dataset):
    """
    Drug-Target Interaction dataset.
    
    Supports loading and preprocessing of:
    - Davis dataset
    - KIBA dataset
    - BindingDB dataset
    
    Args:
        root (str): Root directory containing the dataset
        dataset_name (str): Name of dataset ('davis', 'kiba', 'bindingdb')
        split (str): Data split ('train', 'val', 'test')
        split_type (str): Type of split ('random', 'cold_drug', 'cold_target', 'cold_both')
        transform (callable, optional): Transform to apply to samples
    """
    
    def __init__(
        self,
        root,
        dataset_name='davis',
        split='train',
        split_type='random',
        transform=None,
        cold_start_config=None
    ):
        super(DTIDataset, self).__init__()
        
        self.root = root
        self.dataset_name = dataset_name.lower()
        self.split = split
        self.split_type = split_type
        self.transform = transform
        
        # Load dataset
        self.data = self._load_data()
        
        # Apply split
        if cold_start_config is not None:
            self.data = self._apply_cold_start_split(cold_start_config)
        else:
            self.data = self._apply_split()
    
    def _load_data(self):
        """Load dataset from files."""
        dataset_path = os.path.join(self.root, self.dataset_name.capitalize())
        
        if not os.path.exists(dataset_path):
            raise ValueError(f"Dataset not found at {dataset_path}")
        
        # Check for processed data
        processed_file = os.path.join(dataset_path, 'processed_data.pkl')
        if os.path.exists(processed_file):
            with open(processed_file, 'rb') as f:
                return pickle.load(f)
        
        # Load raw data based on dataset type
        if self.dataset_name == 'davis':
            return self._load_davis()
        elif self.dataset_name == 'kiba':
            return self._load_kiba()
        elif self.dataset_name == 'bindingdb':
            return self._load_bindingdb()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
    
    def _load_davis(self):
        """Load Davis dataset."""
        # Davis dataset contains 68 drugs and 442 proteins
        # with Kd values as binding affinities
        dataset_path = os.path.join(self.root, 'Davis')
        
        data = {
            'drugs': [],
            'proteins': [],
            'affinities': [],
            'drug_smiles': [],
            'protein_sequences': []
        }
        
        # Check if data files exist
        affinity_file = os.path.join(dataset_path, 'affinities.csv')
        if os.path.exists(affinity_file):
            df = pd.read_csv(affinity_file)
            data['drugs'] = df['drug_id'].tolist()
            data['proteins'] = df['protein_id'].tolist()
            data['affinities'] = df['affinity'].tolist()
            
            if 'smiles' in df.columns:
                data['drug_smiles'] = df['smiles'].tolist()
            if 'sequence' in df.columns:
                data['protein_sequences'] = df['sequence'].tolist()
        
        return data
    
    def _load_kiba(self):
        """Load KIBA dataset."""
        # KIBA dataset contains 2,111 drugs and 229 proteins
        # with KIBA scores as binding affinities
        dataset_path = os.path.join(self.root, 'KIBA')
        
        data = {
            'drugs': [],
            'proteins': [],
            'affinities': [],
            'drug_smiles': [],
            'protein_sequences': []
        }
        
        affinity_file = os.path.join(dataset_path, 'affinities.csv')
        if os.path.exists(affinity_file):
            df = pd.read_csv(affinity_file)
            data['drugs'] = df['drug_id'].tolist()
            data['proteins'] = df['protein_id'].tolist()
            data['affinities'] = df['kiba_score'].tolist()
            
            if 'smiles' in df.columns:
                data['drug_smiles'] = df['smiles'].tolist()
            if 'sequence' in df.columns:
                data['protein_sequences'] = df['sequence'].tolist()
        
        return data
    
    def _load_bindingdb(self):
        """Load BindingDB dataset."""
        dataset_path = os.path.join(self.root, 'BindingDB')
        
        data = {
            'drugs': [],
            'proteins': [],
            'affinities': [],
            'drug_smiles': [],
            'protein_sequences': [],
            'binding_types': []
        }
        
        affinity_file = os.path.join(dataset_path, 'affinities.csv')
        if os.path.exists(affinity_file):
            df = pd.read_csv(affinity_file)
            data['drugs'] = df['drug_id'].tolist()
            data['proteins'] = df['protein_id'].tolist()
            data['affinities'] = df['affinity'].tolist()
            
            if 'smiles' in df.columns:
                data['drug_smiles'] = df['smiles'].tolist()
            if 'sequence' in df.columns:
                data['protein_sequences'] = df['sequence'].tolist()
            if 'binding_type' in df.columns:
                data['binding_types'] = df['binding_type'].tolist()
        
        return data
    
    def _apply_split(self):
        """Apply random train/val/test split."""
        # Placeholder for split logic
        # In practice, you would implement proper stratified splitting
        return self.data
    
    def _apply_cold_start_split(self, config):
        """Apply cold-start split based on configuration."""
        # Placeholder for cold-start split logic
        return self.data
    
    def __len__(self):
        """Return dataset size."""
        if len(self.data['drugs']) == 0:
            return 0
        return len(self.data['drugs'])
    
    def __getitem__(self, idx):
        """Get a single sample."""
        sample = {
            'drug_id': self.data['drugs'][idx] if idx < len(self.data['drugs']) else '',
            'protein_id': self.data['proteins'][idx] if idx < len(self.data['proteins']) else '',
            'affinity': self.data['affinities'][idx] if idx < len(self.data['affinities']) else 0.0
        }
        
        if 'drug_smiles' in self.data and idx < len(self.data['drug_smiles']):
            sample['smiles'] = self.data['drug_smiles'][idx]
        
        if 'protein_sequences' in self.data and idx < len(self.data['protein_sequences']):
            sample['sequence'] = self.data['protein_sequences'][idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


class ColdStartSplit:
    """
    Cold-start split generator for DTI datasets.
    
    Implements various cold-start scenarios:
    - Cold drug: Test on drugs not seen during training
    - Cold target: Test on proteins not seen during training
    - Cold both: Test on drug-protein pairs where both are unseen
    
    Args:
        dataset (DTIDataset): DTI dataset to split
        split_type (str): Type of cold-start split
        test_ratio (float): Ratio of test set
        val_ratio (float): Ratio of validation set
        random_seed (int): Random seed for reproducibility
    """
    
    def __init__(
        self,
        dataset,
        split_type='cold_drug',
        test_ratio=0.2,
        val_ratio=0.1,
        random_seed=42
    ):
        self.dataset = dataset
        self.split_type = split_type
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.random_seed = random_seed
        
        np.random.seed(random_seed)
        
    def generate_split(self):
        """Generate cold-start split indices."""
        if self.split_type == 'cold_drug':
            return self._cold_drug_split()
        elif self.split_type == 'cold_target':
            return self._cold_target_split()
        elif self.split_type == 'cold_both':
            return self._cold_both_split()
        elif self.split_type == 'random':
            return self._random_split()
        else:
            raise ValueError(f"Unknown split type: {self.split_type}")
    
    def _cold_drug_split(self):
        """Generate cold drug split."""
        drugs = np.unique(self.dataset.data['drugs'])
        n_drugs = len(drugs)
        
        # Split drugs
        n_test = int(n_drugs * self.test_ratio)
        n_val = int(n_drugs * self.val_ratio)
        
        shuffled_drugs = np.random.permutation(drugs)
        test_drugs = set(shuffled_drugs[:n_test])
        val_drugs = set(shuffled_drugs[n_test:n_test+n_val])
        train_drugs = set(shuffled_drugs[n_test+n_val:])
        
        # Get indices
        train_idx = [i for i, d in enumerate(self.dataset.data['drugs']) if d in train_drugs]
        val_idx = [i for i, d in enumerate(self.dataset.data['drugs']) if d in val_drugs]
        test_idx = [i for i, d in enumerate(self.dataset.data['drugs']) if d in test_drugs]
        
        return {
            'train': train_idx,
            'val': val_idx,
            'test': test_idx,
            'test_drugs': list(test_drugs),
            'val_drugs': list(val_drugs)
        }
    
    def _cold_target_split(self):
        """Generate cold target split."""
        proteins = np.unique(self.dataset.data['proteins'])
        n_proteins = len(proteins)
        
        # Split proteins
        n_test = int(n_proteins * self.test_ratio)
        n_val = int(n_proteins * self.val_ratio)
        
        shuffled_proteins = np.random.permutation(proteins)
        test_proteins = set(shuffled_proteins[:n_test])
        val_proteins = set(shuffled_proteins[n_test:n_test+n_val])
        train_proteins = set(shuffled_proteins[n_test+n_val:])
        
        # Get indices
        train_idx = [i for i, p in enumerate(self.dataset.data['proteins']) if p in train_proteins]
        val_idx = [i for i, p in enumerate(self.dataset.data['proteins']) if p in val_proteins]
        test_idx = [i for i, p in enumerate(self.dataset.data['proteins']) if p in test_proteins]
        
        return {
            'train': train_idx,
            'val': val_idx,
            'test': test_idx,
            'test_proteins': list(test_proteins),
            'val_proteins': list(val_proteins)
        }
    
    def _cold_both_split(self):
        """Generate cold both split (unseen drug-protein pairs)."""
        drug_split = self._cold_drug_split()
        protein_split = self._cold_target_split()
        
        test_drugs = set(drug_split['test_drugs'])
        test_proteins = set(protein_split['test_proteins'])
        
        # Test set: both drug and protein are unseen
        test_idx = [
            i for i, (d, p) in enumerate(zip(self.dataset.data['drugs'], self.dataset.data['proteins']))
            if d in test_drugs and p in test_proteins
        ]
        
        # Train set: both drug and protein are seen
        train_drugs = set(self.dataset.data['drugs']) - test_drugs
        train_proteins = set(self.dataset.data['proteins']) - test_proteins
        train_idx = [
            i for i, (d, p) in enumerate(zip(self.dataset.data['drugs'], self.dataset.data['proteins']))
            if d in train_drugs and p in train_proteins
        ]
        
        # Validation set: one seen, one unseen
        all_idx = set(range(len(self.dataset.data['drugs'])))
        remaining_idx = list(all_idx - set(train_idx) - set(test_idx))
        val_idx = remaining_idx[:int(len(remaining_idx) * 0.5)]
        
        return {
            'train': train_idx,
            'val': val_idx,
            'test': test_idx
        }
    
    def _random_split(self):
        """Generate random split."""
        n_samples = len(self.dataset)
        indices = np.random.permutation(n_samples)
        
        n_test = int(n_samples * self.test_ratio)
        n_val = int(n_samples * self.val_ratio)
        
        test_idx = indices[:n_test].tolist()
        val_idx = indices[n_test:n_test+n_val].tolist()
        train_idx = indices[n_test+n_val:].tolist()
        
        return {
            'train': train_idx,
            'val': val_idx,
            'test': test_idx
        }
    
    def save_split(self, save_path):
        """Save split indices to file."""
        split_info = self.generate_split()
        
        with open(save_path, 'w') as f:
            json.dump(split_info, f, indent=2)
        
        return split_info
    
    def load_split(self, load_path):
        """Load split indices from file."""
        with open(load_path, 'r') as f:
            split_info = json.load(f)
        
        return split_info
