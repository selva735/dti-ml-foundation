"""Cold-start scenario evaluation for DTI prediction.

This module provides utilities for evaluating models in cold-start scenarios
where test data contains unseen drugs, targets, or both.
"""

from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics import jaccard_score


class ColdStartEvaluator:
    """Evaluator for cold-start scenarios.
    
    Handles evaluation when test set contains:
    - Unseen drugs (new chemical compounds)
    - Unseen targets (new proteins)
    - Unseen drug-target pairs
    """
    
    def __init__(self):
        """Initialize cold-start evaluator."""
        pass
    
    def identify_cold_start_samples(
        self,
        test_drugs: List[str],
        test_targets: List[str],
        train_drugs: Set[str],
        train_targets: Set[str],
    ) -> Dict[str, np.ndarray]:
        """Identify samples in different cold-start categories.
        
        Args:
            test_drugs: List of drug SMILES in test set
            test_targets: List of protein sequences in test set
            train_drugs: Set of drug SMILES in training set
            train_targets: Set of protein sequences in training set
            
        Returns:
            Dictionary with boolean masks for each cold-start category
        """
        test_drugs = np.array(test_drugs)
        test_targets = np.array(test_targets)
        
        # Identify different scenarios
        unseen_drug_mask = np.array([d not in train_drugs for d in test_drugs])
        unseen_target_mask = np.array([t not in train_targets for t in test_targets])
        
        seen_both_mask = (~unseen_drug_mask) & (~unseen_target_mask)
        unseen_drug_only_mask = unseen_drug_mask & (~unseen_target_mask)
        unseen_target_only_mask = (~unseen_drug_mask) & unseen_target_mask
        unseen_both_mask = unseen_drug_mask & unseen_target_mask
        
        return {
            'seen_both': seen_both_mask,
            'unseen_drug': unseen_drug_only_mask,
            'unseen_target': unseen_target_only_mask,
            'unseen_both': unseen_both_mask,
        }
    
    def evaluate_by_scenario(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        scenario_masks: Dict[str, np.ndarray],
        evaluator,
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate model separately for each cold-start scenario.
        
        Args:
            predictions: Model predictions
            targets: Ground truth values
            scenario_masks: Dictionary of scenario masks
            evaluator: Evaluator object for computing metrics
            
        Returns:
            Dictionary mapping scenario to metrics
        """
        results = {}
        
        for scenario_name, mask in scenario_masks.items():
            n_samples = mask.sum()
            
            if n_samples == 0:
                continue
            
            scenario_preds = predictions[mask]
            scenario_targets = targets[mask]
            
            metrics = evaluator.compute_metrics(scenario_preds, scenario_targets)
            metrics['n_samples'] = int(n_samples)
            
            results[scenario_name] = metrics
        
        return results
    
    def compute_drug_similarity(
        self,
        drug1_smiles: str,
        drug2_smiles: str,
        method: str = "tanimoto",
    ) -> float:
        """Compute similarity between two drugs.
        
        Args:
            drug1_smiles: SMILES for first drug
            drug2_smiles: SMILES for second drug
            method: Similarity method ('tanimoto', 'dice')
            
        Returns:
            Similarity score between 0 and 1
        """
        mol1 = Chem.MolFromSmiles(drug1_smiles)
        mol2 = Chem.MolFromSmiles(drug2_smiles)
        
        if mol1 is None or mol2 is None:
            return 0.0
        
        # Compute Morgan fingerprints
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=2, nBits=2048)
        
        # Convert to arrays
        arr1 = np.zeros((1,))
        arr2 = np.zeros((1,))
        
        if method == "tanimoto":
            from rdkit import DataStructs
            return DataStructs.TanimotoSimilarity(fp1, fp2)
        elif method == "dice":
            from rdkit import DataStructs
            return DataStructs.DiceSimilarity(fp1, fp2)
        else:
            raise ValueError(f"Unknown similarity method: {method}")
    
    def evaluate_by_similarity(
        self,
        test_drugs: List[str],
        predictions: np.ndarray,
        targets: np.ndarray,
        train_drugs: List[str],
        evaluator,
        similarity_thresholds: List[float] = [0.3, 0.5, 0.7, 0.9],
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate by drug similarity to training set.
        
        Args:
            test_drugs: Test set drug SMILES
            predictions: Model predictions
            targets: Ground truth values
            train_drugs: Training set drug SMILES
            evaluator: Evaluator for metrics
            similarity_thresholds: Thresholds for grouping by similarity
            
        Returns:
            Dictionary mapping similarity range to metrics
        """
        # Compute maximum similarity to training set for each test drug
        max_similarities = []
        
        for test_drug in test_drugs:
            similarities = []
            for train_drug in train_drugs:
                sim = self.compute_drug_similarity(test_drug, train_drug)
                similarities.append(sim)
            
            if len(similarities) > 0:
                max_similarities.append(max(similarities))
            else:
                max_similarities.append(0.0)
        
        max_similarities = np.array(max_similarities)
        
        # Evaluate by similarity ranges
        results = {}
        thresholds = [0.0] + similarity_thresholds + [1.0]
        
        for i in range(len(thresholds) - 1):
            lower = thresholds[i]
            upper = thresholds[i + 1]
            
            mask = (max_similarities >= lower) & (max_similarities < upper)
            n_samples = mask.sum()
            
            if n_samples == 0:
                continue
            
            range_preds = predictions[mask]
            range_targets = targets[mask]
            
            metrics = evaluator.compute_metrics(range_preds, range_targets)
            metrics['n_samples'] = int(n_samples)
            metrics['mean_similarity'] = max_similarities[mask].mean()
            
            range_name = f"sim_{lower:.1f}-{upper:.1f}"
            results[range_name] = metrics
        
        return results


class FewShotAdapter:
    """Adapter for few-shot learning in cold-start scenarios.
    
    Provides methods for adapting models with limited data from new drugs/targets.
    """
    
    def __init__(
        self,
        model,
        learning_rate: float = 1e-4,
        n_epochs: int = 10,
    ):
        """Initialize few-shot adapter.
        
        Args:
            model: Model to adapt
            learning_rate: Learning rate for adaptation
            n_epochs: Number of adaptation epochs
        """
        self.model = model
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
    
    def adapt(
        self,
        support_loader,
        device: str = 'cuda',
        freeze_encoder: bool = True,
    ):
        """Adapt model on support set.
        
        Args:
            support_loader: DataLoader with support samples
            device: Device to run on
            freeze_encoder: Whether to freeze encoder layers
        """
        if freeze_encoder:
            # Freeze encoder parameters
            for param in self.model.drug_encoder.parameters():
                param.requires_grad = False
            for param in self.model.protein_encoder.parameters():
                param.requires_grad = False
        
        # Set up optimizer
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.learning_rate,
        )
        
        criterion = torch.nn.MSELoss()
        
        # Adaptation loop
        self.model.train()
        
        for epoch in range(self.n_epochs):
            total_loss = 0.0
            
            for batch in support_loader:
                drug_graph = batch['drug_graph'].to(device)
                protein_emb = batch['protein_emb'].to(device)
                affinity = batch['affinity'].to(device)
                
                optimizer.zero_grad()
                
                predictions, _ = self.model(drug_graph, protein_emb)
                loss = criterion(predictions, affinity)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
        
        # Unfreeze parameters
        if freeze_encoder:
            for param in self.model.parameters():
                param.requires_grad = True


def print_cold_start_results(
    results: Dict[str, Dict[str, float]],
    metric: str = 'rmse',
):
    """Print cold-start evaluation results in a formatted way.
    
    Args:
        results: Dictionary mapping scenario to metrics
        metric: Primary metric to highlight
    """
    print("\nCold-Start Evaluation Results:")
    print("=" * 70)
    print(f"{'Scenario':<20} {'N Samples':<12} {metric.upper():<10} {'Pearson':<10}")
    print("-" * 70)
    
    for scenario, metrics in results.items():
        n_samples = metrics.get('n_samples', 0)
        metric_val = metrics.get(metric, 0.0)
        pearson = metrics.get('pearson', 0.0)
        
        print(f"{scenario:<20} {n_samples:<12} {metric_val:<10.4f} {pearson:<10.4f}")
    
    print("=" * 70)
