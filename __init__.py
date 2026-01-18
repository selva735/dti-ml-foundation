"""
DTI-ML-Foundation: Multi-modal Drug-Target Interaction Prediction
"""

__version__ = "0.1.0"
__author__ = "DTI-ML-Foundation Team"
__license__ = "MIT"

from models import DrugGNN, ProteinPLM, TDAEncoder, FusionAttention, EvidentialHead
from data import DTIDataset, ColdStartSplit

__all__ = [
    'DrugGNN',
    'ProteinPLM',
    'TDAEncoder',
    'FusionAttention',
    'EvidentialHead',
    'DTIDataset',
    'ColdStartSplit'
]
