"""CMAT-DTI: Cross-Modal Attention Transformer for Drug-Target Interaction.

Main model class combining:
  - MolecularGraphTransformer (drug encoder)
  - ProteinSequenceTransformer (protein encoder)
  - BidirectionalCrossAttention (fusion)
  - Prediction head with optional uncertainty estimation (MC Dropout)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .cross_attention import BidirectionalCrossAttention
from .drug_encoder import MolecularGraphTransformer
from .protein_encoder import ProteinSequenceTransformer


@dataclass
class ModelConfig:
    """Hyperparameter configuration for CMAT-DTI.

    All parameters have sensible defaults that work well on standard DTI
    benchmarks (Davis, KIBA, BindingDB).
    """

    # Drug encoder
    atom_feat_dim: int = 78
    drug_embed_dim: int = 256
    drug_num_layers: int = 4
    drug_num_heads: int = 8
    drug_ffn_dim: int = 512
    drug_max_distance: int = 10
    drug_dropout: float = 0.1
    drug_pooling: str = "attention"

    # Protein encoder
    protein_embed_dim: int = 256
    protein_num_layers: int = 4
    protein_num_heads: int = 8
    protein_ffn_dim: int = 512
    protein_max_seq_len: int = 1002
    protein_dropout: float = 0.1
    plm_input_dim: Optional[int] = None  # Set if using pre-trained PLM embeddings

    # Cross-modal attention fusion
    cross_dim: int = 256
    cross_num_heads: int = 8
    cross_num_layers: int = 2
    cross_dropout: float = 0.1
    fusion_output_dim: int = 512

    # Prediction head
    head_hidden_dims: list = field(default_factory=lambda: [256, 128])
    head_dropout: float = 0.2
    output_dim: int = 1  # 1 for regression, >1 for classification

    # Uncertainty
    mc_dropout_samples: int = 20
    enable_uncertainty: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        """Create config from a dictionary (e.g. loaded from YAML)."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class PredictionHead(nn.Module):
    """MLP prediction head with residual connections and dropout.

    Args:
        input_dim: Dimensionality of the fused representation.
        hidden_dims: Sizes of hidden layers.
        output_dim: Number of output units.
        dropout: Dropout probability (kept active during training and MC
            dropout inference).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        dims = [input_dim] + hidden_dims
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers += [
                nn.Linear(dims[i], dims[i + 1]),
                nn.LayerNorm(dims[i + 1]),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
        self.mlp = nn.Sequential(*layers)
        self.output_layer = nn.Linear(dims[-1], output_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.output_layer(self.mlp(x))


class CMATDTI(nn.Module):
    """Cross-Modal Attention Transformer for Drug-Target Interaction prediction.

    This model implements a novel DTI prediction approach with three key
    innovations over prior GNN-based methods:

    1. **Molecular Graph Transformer** – extends standard GNN message-passing
       with transformer-style attention biased by graph-structural distances,
       allowing the model to capture long-range atom interactions missed by
       shallow GNN layers.

    2. **Bidirectional Cross-Modal Attention** – unlike unidirectional
       cross-attention (drug → protein), both modalities simultaneously
       attend to each other, enabling richer information exchange and
       context-aware representations.

    3. **Monte Carlo Dropout uncertainty estimation** – at inference time the
       model can produce calibrated prediction intervals by running multiple
       stochastic forward passes, enabling reliable uncertainty quantification
       useful for drug discovery prioritisation.

    Args:
        config: :class:`ModelConfig` instance controlling all hyperparameters.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.drug_encoder = MolecularGraphTransformer(
            atom_feat_dim=config.atom_feat_dim,
            embed_dim=config.drug_embed_dim,
            num_layers=config.drug_num_layers,
            num_heads=config.drug_num_heads,
            ffn_dim=config.drug_ffn_dim,
            max_distance=config.drug_max_distance,
            dropout=config.drug_dropout,
            pooling=config.drug_pooling,
            output_dim=config.cross_dim,
        )

        self.protein_encoder = ProteinSequenceTransformer(
            embed_dim=config.protein_embed_dim,
            num_layers=config.protein_num_layers,
            num_heads=config.protein_num_heads,
            ffn_dim=config.protein_ffn_dim,
            max_seq_len=config.protein_max_seq_len,
            dropout=config.protein_dropout,
            output_dim=config.cross_dim,
            plm_input_dim=config.plm_input_dim,
        )

        self.fusion = BidirectionalCrossAttention(
            drug_dim=config.cross_dim,
            protein_dim=config.cross_dim,
            cross_dim=config.cross_dim,
            num_heads=config.cross_num_heads,
            num_layers=config.cross_num_layers,
            dropout=config.cross_dropout,
            output_dim=config.fusion_output_dim,
        )

        self.head = PredictionHead(
            input_dim=config.fusion_output_dim,
            hidden_dims=config.head_hidden_dims,
            output_dim=config.output_dim,
            dropout=config.head_dropout,
        )

    def forward(
        self,
        atom_features: Tensor,
        dist_matrix: Tensor,
        protein_tokens: Optional[Tensor] = None,
        atom_mask: Optional[Tensor] = None,
        protein_padding_mask: Optional[Tensor] = None,
        plm_embeddings: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass returning predicted binding affinity.

        Args:
            atom_features: ``(batch, max_atoms, atom_feat_dim)``
            dist_matrix: ``(batch, max_atoms, max_atoms)``
            protein_tokens: ``(batch, seq_len)`` – required when
                *plm_embeddings* is None.
            atom_mask: ``(batch, max_atoms)`` bool, True = padding atom.
            protein_padding_mask: ``(batch, seq_len)`` bool, True = padding.
            plm_embeddings: ``(batch, seq_len, plm_input_dim)`` – optional
                pre-computed PLM embeddings.

        Returns:
            Predicted binding affinity of shape ``(batch, output_dim)``.
        """
        drug_repr = self.drug_encoder(atom_features, dist_matrix, atom_mask)
        protein_repr = self.protein_encoder(
            tokens=protein_tokens,
            padding_mask=protein_padding_mask,
            plm_embeddings=plm_embeddings,
        )
        fused, _, _ = self.fusion(drug_repr, protein_repr)
        return self.head(fused)

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        atom_features: Tensor,
        dist_matrix: Tensor,
        protein_tokens: Optional[Tensor] = None,
        atom_mask: Optional[Tensor] = None,
        protein_padding_mask: Optional[Tensor] = None,
        plm_embeddings: Optional[Tensor] = None,
        n_samples: Optional[int] = None,
    ) -> Dict[str, Tensor]:
        """Estimate predictions and uncertainty via Monte Carlo Dropout.

        Dropout layers are kept active during inference; multiple stochastic
        forward passes are averaged to obtain a mean prediction and variance.

        Args:
            *: Same positional/keyword arguments as :meth:`forward`.
            n_samples: Number of MC samples (default:
                ``config.mc_dropout_samples``).

        Returns:
            Dict with keys:
              - ``"mean"`` – mean prediction ``(batch, output_dim)``
              - ``"std"`` – standard deviation ``(batch, output_dim)``
              - ``"samples"`` – all samples ``(n_samples, batch, output_dim)``
        """
        if n_samples is None:
            n_samples = self.config.mc_dropout_samples

        # Enable dropout for stochastic inference
        self.train()  # activates dropout
        samples = []
        for _ in range(n_samples):
            pred = self.forward(
                atom_features,
                dist_matrix,
                protein_tokens=protein_tokens,
                atom_mask=atom_mask,
                protein_padding_mask=protein_padding_mask,
                plm_embeddings=plm_embeddings,
            )
            samples.append(pred)

        self.eval()
        stacked = torch.stack(samples, dim=0)  # (n_samples, batch, output_dim)
        return {
            "mean": stacked.mean(dim=0),
            "std": stacked.std(dim=0),
            "samples": stacked,
        }


def create_model(config: Optional[ModelConfig] = None, **kwargs) -> CMATDTI:
    """Convenience factory to create a :class:`CMATDTI` model.

    Args:
        config: Optional :class:`ModelConfig`.  If None, a default config is
            created and any *kwargs* are applied to it.
        **kwargs: Keyword arguments forwarded to :class:`ModelConfig` when
            *config* is None.

    Returns:
        Initialised :class:`CMATDTI` model.
    """
    if config is None:
        config = ModelConfig(**kwargs)
    return CMATDTI(config)
