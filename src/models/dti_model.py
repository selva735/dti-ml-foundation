"""Main DTI prediction model.

This module implements the complete drug-target interaction prediction model
combining graph neural networks, protein embeddings, and attention mechanisms.
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gnn_layers import GraphEncoder
from .attention import CrossModalAttention, SelfAttention


class DTIModel(nn.Module):
    """Drug-Target Interaction prediction model.
    
    Multi-modal attention-based GNN framework that combines:
    - Drug molecular graph encoding via GNN
    - Protein PLM embeddings
    - Cross-modal attention for interaction modeling
    - Prediction head for affinity estimation
    """
    
    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        protein_emb_dim: int,
        hidden_dim: int = 256,
        gnn_layers: int = 3,
        gnn_type: str = "gat",
        gnn_heads: int = 4,
        attention_heads: int = 8,
        dropout: float = 0.2,
        pooling: str = "mean",
        use_cross_attention: bool = True,
        use_self_attention: bool = True,
    ):
        """Initialize DTI model.
        
        Args:
            node_feat_dim: Molecular graph node feature dimension
            edge_feat_dim: Molecular graph edge feature dimension
            protein_emb_dim: Protein embedding dimension
            hidden_dim: Hidden dimension for encoders
            gnn_layers: Number of GNN layers
            gnn_type: Type of GNN ('gat' or 'gin')
            gnn_heads: Number of attention heads for GAT
            attention_heads: Number of heads for cross-modal attention
            dropout: Dropout rate
            pooling: Graph pooling strategy ('mean', 'max', 'add')
            use_cross_attention: Whether to use cross-modal attention
            use_self_attention: Whether to use self-attention
        """
        super().__init__()
        
        self.use_cross_attention = use_cross_attention
        self.use_self_attention = use_self_attention
        
        # Drug encoder (GNN for molecular graphs)
        self.drug_encoder = GraphEncoder(
            node_feat_dim=node_feat_dim,
            hidden_dim=hidden_dim,
            num_layers=gnn_layers,
            gnn_type=gnn_type,
            heads=gnn_heads,
            dropout=dropout,
            edge_feat_dim=edge_feat_dim,
            pooling=pooling,
        )
        
        drug_output_dim = self.drug_encoder.output_dim
        
        # Protein encoder (projection layer for PLM embeddings)
        self.protein_encoder = nn.Sequential(
            nn.Linear(protein_emb_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        protein_output_dim = hidden_dim
        
        # Cross-modal attention
        if use_cross_attention:
            self.cross_attention = CrossModalAttention(
                drug_dim=drug_output_dim,
                protein_dim=protein_output_dim,
                hidden_dim=hidden_dim,
                num_heads=attention_heads,
                dropout=dropout,
            )
            fusion_dim = hidden_dim * 2  # Concatenate drug and protein contexts
        else:
            fusion_dim = drug_output_dim + protein_output_dim
        
        # Self-attention for feature refinement
        if use_self_attention:
            self.self_attention = SelfAttention(
                input_dim=fusion_dim,
                num_heads=attention_heads,
                dropout=dropout,
            )
        
        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(
        self,
        drug_graph,
        protein_emb: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """Forward pass.
        
        Args:
            drug_graph: PyTorch Geometric Batch object with molecular graphs
            protein_emb: Protein embeddings [batch_size, protein_emb_dim]
            return_attention: Whether to return attention weights
            
        Returns:
            Tuple of (predictions, attention_weights)
            predictions: [batch_size, 1]
            attention_weights: Dictionary of attention weights (if return_attention=True)
        """
        # Encode drug molecular graph
        drug_emb = self.drug_encoder(
            drug_graph.x,
            drug_graph.edge_index,
            drug_graph.edge_attr if hasattr(drug_graph, 'edge_attr') else None,
            drug_graph.batch,
        )
        
        # Encode protein
        protein_enc = self.protein_encoder(protein_emb)
        
        # Cross-modal attention
        attention_weights = {}
        if self.use_cross_attention:
            drug_context, protein_context, cross_attn = self.cross_attention(
                drug_emb, protein_enc, return_attention=return_attention
            )
            if return_attention and cross_attn is not None:
                attention_weights['cross_modal'] = cross_attn
            
            # Fuse drug and protein contexts
            fusion = torch.cat([drug_context, protein_context], dim=1)
        else:
            # Simple concatenation
            fusion = torch.cat([drug_emb, protein_enc], dim=1)
        
        # Self-attention for refinement
        if self.use_self_attention:
            fusion, self_attn = self.self_attention(
                fusion, return_attention=return_attention
            )
            if return_attention and self_attn is not None:
                attention_weights['self_attention'] = self_attn
        
        # Predict affinity
        predictions = self.predictor(fusion)
        
        if return_attention:
            return predictions, attention_weights
        else:
            return predictions, None
    
    def get_embeddings(
        self,
        drug_graph,
        protein_emb: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Get intermediate embeddings for visualization/analysis.
        
        Args:
            drug_graph: PyTorch Geometric Batch object with molecular graphs
            protein_emb: Protein embeddings [batch_size, protein_emb_dim]
            
        Returns:
            Dictionary with embeddings at different stages
        """
        # Encode drug and protein
        drug_emb = self.drug_encoder(
            drug_graph.x,
            drug_graph.edge_index,
            drug_graph.edge_attr if hasattr(drug_graph, 'edge_attr') else None,
            drug_graph.batch,
        )
        protein_enc = self.protein_encoder(protein_emb)
        
        embeddings = {
            'drug_embedding': drug_emb,
            'protein_embedding': protein_enc,
        }
        
        # Cross-modal attention
        if self.use_cross_attention:
            drug_context, protein_context, _ = self.cross_attention(
                drug_emb, protein_enc, return_attention=False
            )
            embeddings['drug_context'] = drug_context
            embeddings['protein_context'] = protein_context
            fusion = torch.cat([drug_context, protein_context], dim=1)
        else:
            fusion = torch.cat([drug_emb, protein_enc], dim=1)
        
        embeddings['fusion'] = fusion
        
        return embeddings


def create_dti_model(
    feature_dims: Dict[str, int],
    config: Optional[Dict] = None,
) -> DTIModel:
    """Factory function to create DTI model from configuration.
    
    Args:
        feature_dims: Dictionary with feature dimensions
        config: Model configuration dictionary
        
    Returns:
        DTI model instance
    """
    if config is None:
        config = {}
    
    return DTIModel(
        node_feat_dim=feature_dims['node_feat_dim'],
        edge_feat_dim=feature_dims['edge_feat_dim'],
        protein_emb_dim=feature_dims['protein_emb_dim'],
        hidden_dim=config.get('hidden_dim', 256),
        gnn_layers=config.get('gnn_layers', 3),
        gnn_type=config.get('gnn_type', 'gat'),
        gnn_heads=config.get('gnn_heads', 4),
        attention_heads=config.get('attention_heads', 8),
        dropout=config.get('dropout', 0.2),
        pooling=config.get('pooling', 'mean'),
        use_cross_attention=config.get('use_cross_attention', True),
        use_self_attention=config.get('use_self_attention', True),
    )
