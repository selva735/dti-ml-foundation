"""Tests for data processing modules."""

import pytest
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data.molecular_graph import MolecularGraphGenerator, smiles_to_graph
from data.preprocessing import DTIPreprocessor


class TestMolecularGraphGenerator:
    """Tests for molecular graph generation."""
    
    def test_smiles_to_graph_valid(self):
        """Test conversion of valid SMILES to graph."""
        smiles = "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"  # Ibuprofen
        
        graph = smiles_to_graph(smiles)
        
        assert graph is not None
        assert hasattr(graph, 'x')
        assert hasattr(graph, 'edge_index')
        assert graph.x.size(0) > 0  # Has nodes
    
    def test_smiles_to_graph_invalid(self):
        """Test handling of invalid SMILES."""
        smiles = "INVALID_SMILES"
        
        graph = smiles_to_graph(smiles)
        
        assert graph is None
    
    def test_feature_dimensions(self):
        """Test that feature dimensions are consistent."""
        generator = MolecularGraphGenerator()
        
        node_dim = generator.get_node_feature_dim()
        edge_dim = generator.get_edge_feature_dim()
        
        assert node_dim > 0
        assert edge_dim > 0
        
        # Test with a SMILES
        smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
        graph = generator.smiles_to_graph(smiles)
        
        assert graph.x.size(1) == node_dim
        if graph.edge_attr.size(0) > 0:
            assert graph.edge_attr.size(1) == edge_dim


class TestDTIPreprocessor:
    """Tests for data preprocessing."""
    
    def test_clean_data(self):
        """Test data cleaning."""
        # Create sample data with issues
        df = pd.DataFrame({
            'smiles': ['CC', 'CCC', 'CC', None, 'CCCC'],
            'sequence': ['AAA', 'BBB', 'AAA', 'CCC', 'DDD'],
            'affinity': [5.0, 6.0, 5.0, 7.0, 8.0],
        })
        
        preprocessor = DTIPreprocessor()
        df_clean = preprocessor.clean_data(df, remove_duplicates=True, remove_missing=True)
        
        # Should remove None and duplicate
        assert len(df_clean) < len(df)
        assert df_clean['smiles'].isna().sum() == 0
    
    def test_split_data(self):
        """Test data splitting."""
        # Create sample data
        df = pd.DataFrame({
            'smiles': ['CC'] * 100,
            'sequence': ['AAA'] * 100,
            'affinity': np.random.uniform(4, 10, 100),
        })
        
        preprocessor = DTIPreprocessor()
        train_df, val_df, test_df = preprocessor.split_data(
            df, test_size=0.2, val_size=0.1, random_state=42
        )
        
        # Check splits
        total = len(train_df) + len(val_df) + len(test_df)
        assert total == len(df)
        assert len(test_df) / len(df) == pytest.approx(0.2, rel=0.1)
    
    def test_normalize_targets(self):
        """Test target normalization."""
        df = pd.DataFrame({
            'smiles': ['CC'] * 10,
            'sequence': ['AAA'] * 10,
            'affinity': np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float),
        })
        
        preprocessor = DTIPreprocessor()
        df_norm = preprocessor.normalize_targets(df, method='standardize', fit=True)
        
        # Check normalization
        mean = df_norm['affinity'].mean()
        std = df_norm['affinity'].std()
        
        assert mean == pytest.approx(0.0, abs=0.1)
        assert std == pytest.approx(1.0, abs=0.1)
    
    def test_cold_start_split(self):
        """Test cold-start splitting."""
        # Create sample data with different drugs and targets
        drugs = ['CC', 'CCC', 'CCCC', 'CCCCC', 'CCCCCC'] * 20
        targets = ['AAA', 'BBB', 'CCC', 'DDD', 'EEE'] * 20
        
        df = pd.DataFrame({
            'smiles': drugs,
            'sequence': targets,
            'affinity': np.random.uniform(4, 10, 100),
        })
        
        preprocessor = DTIPreprocessor()
        train_df, val_df, test_df = preprocessor.create_cold_start_split(
            df, mode='drug', test_size=0.2, random_state=42
        )
        
        # Check that test drugs are not in training
        test_drugs = set(test_df['smiles'].unique())
        train_drugs = set(train_df['smiles'].unique())
        
        assert len(test_drugs.intersection(train_drugs)) == 0
