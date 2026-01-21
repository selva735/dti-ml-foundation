"""DTI-ML-Foundation: Multi-modal attention-based GNN framework for drug-target interaction prediction.

This package provides a comprehensive framework for predicting drug-target interactions
using graph neural networks, protein language models, and attention mechanisms.
"""

__version__ = "0.1.0"
__author__ = "DTI-ML-Foundation Contributors"

# Import main classes for convenience
from .data.molecular_graph import MolecularGraphGenerator, smiles_to_graph
from .data.protein_embedding import ProteinEmbeddingGenerator, get_protein_embedding
from .data.preprocessing import DTIPreprocessor
from .data.dataset import DTIDataset, create_dataloader

from .models.dti_model import DTIModel, create_dti_model
from .models.uncertainty import UncertaintyEstimator, MCDropout, EnsembleModel

from .training.trainer import Trainer, create_optimizer, create_scheduler
from .training.evaluator import DTIEvaluator, StratifiedEvaluator
from .training.cold_start import ColdStartEvaluator, FewShotAdapter

from .utils.config import Config, load_config
from .utils.logger import setup_logger, get_logger, TrainingLogger

__all__ = [
    # Data processing
    'MolecularGraphGenerator',
    'smiles_to_graph',
    'ProteinEmbeddingGenerator',
    'get_protein_embedding',
    'DTIPreprocessor',
    'DTIDataset',
    'create_dataloader',
    
    # Models
    'DTIModel',
    'create_dti_model',
    'UncertaintyEstimator',
    'MCDropout',
    'EnsembleModel',
    
    # Training
    'Trainer',
    'create_optimizer',
    'create_scheduler',
    'DTIEvaluator',
    'StratifiedEvaluator',
    'ColdStartEvaluator',
    'FewShotAdapter',
    
    # Utilities
    'Config',
    'load_config',
    'setup_logger',
    'get_logger',
    'TrainingLogger',
]
