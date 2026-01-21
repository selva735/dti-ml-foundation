"""Multi-modal attention mechanisms for DTI prediction.

This module implements attention mechanisms for fusing drug and protein
representations and capturing interaction patterns.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism.
    
    Implements scaled dot-product attention with multiple heads for
    capturing diverse interaction patterns.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """Initialize multi-head attention.
        
        Args:
            embed_dim: Embedding dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.
        
        Args:
            query: Query tensor [batch_size, seq_len_q, embed_dim]
            key: Key tensor [batch_size, seq_len_k, embed_dim]
            value: Value tensor [batch_size, seq_len_v, embed_dim]
            mask: Attention mask [batch_size, seq_len_q, seq_len_k] (optional)
            return_attention: Whether to return attention weights
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size = query.size(0)
        
        # Linear projections and reshape for multi-head
        Q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to values
        context = torch.matmul(attention, V)
        
        # Reshape and project output
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        output = self.out_proj(context)
        
        if return_attention:
            return output, attention
        else:
            return output, None


class CrossModalAttention(nn.Module):
    """Cross-modal attention for drug-protein interaction.
    
    Computes attention between drug and protein representations to
    capture interaction patterns.
    """
    
    def __init__(
        self,
        drug_dim: int,
        protein_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """Initialize cross-modal attention.
        
        Args:
            drug_dim: Drug representation dimension
            protein_dim: Protein representation dimension
            hidden_dim: Hidden dimension for attention
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        # Project drug and protein to common space
        self.drug_proj = nn.Linear(drug_dim, hidden_dim)
        self.protein_proj = nn.Linear(protein_dim, hidden_dim)
        
        # Multi-head attention
        self.attention = MultiHeadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
    
    def forward(
        self,
        drug_emb: torch.Tensor,
        protein_emb: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.
        
        Args:
            drug_emb: Drug embeddings [batch_size, drug_dim]
            protein_emb: Protein embeddings [batch_size, protein_dim]
            return_attention: Whether to return attention weights
            
        Returns:
            Tuple of (drug_context, protein_context, attention_weights)
        """
        # Project to common space
        drug_proj = self.drug_proj(drug_emb).unsqueeze(1)  # [batch, 1, hidden]
        protein_proj = self.protein_proj(protein_emb).unsqueeze(1)  # [batch, 1, hidden]
        
        # Drug attends to protein
        drug_context, attn_d2p = self.attention(
            drug_proj, protein_proj, protein_proj, return_attention=return_attention
        )
        drug_context = self.layer_norm1(drug_proj + drug_context)
        drug_context = self.layer_norm2(drug_context + self.ffn(drug_context))
        
        # Protein attends to drug
        protein_context, attn_p2d = self.attention(
            protein_proj, drug_proj, drug_proj, return_attention=return_attention
        )
        protein_context = self.layer_norm1(protein_proj + protein_context)
        protein_context = self.layer_norm2(protein_context + self.ffn(protein_context))
        
        # Remove sequence dimension
        drug_context = drug_context.squeeze(1)
        protein_context = protein_context.squeeze(1)
        
        # Return attention weights if requested
        attention_weights = None
        if return_attention and attn_d2p is not None:
            attention_weights = {
                'drug_to_protein': attn_d2p,
                'protein_to_drug': attn_p2d,
            }
        
        return drug_context, protein_context, attention_weights


class SelfAttention(nn.Module):
    """Self-attention for feature refinement.
    
    Applies self-attention to refine features before prediction.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """Initialize self-attention.
        
        Args:
            input_dim: Input feature dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.attention = MultiHeadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        self.layer_norm = nn.LayerNorm(input_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, input_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * 4, input_dim),
            nn.Dropout(dropout),
        )
        
        self.layer_norm2 = nn.LayerNorm(input_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.
        
        Args:
            x: Input features [batch_size, input_dim]
            return_attention: Whether to return attention weights
            
        Returns:
            Tuple of (refined_features, attention_weights)
        """
        # Add sequence dimension
        x = x.unsqueeze(1)  # [batch, 1, input_dim]
        
        # Self-attention
        attn_out, attn_weights = self.attention(
            x, x, x, return_attention=return_attention
        )
        x = self.layer_norm(x + attn_out)
        
        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.layer_norm2(x + ffn_out)
        
        # Remove sequence dimension
        x = x.squeeze(1)
        
        return x, attn_weights


class AttentionPooling(nn.Module):
    """Attention-based pooling mechanism.
    
    Learns to weight different features based on their importance.
    """
    
    def __init__(self, input_dim: int):
        """Initialize attention pooling.
        
        Args:
            input_dim: Input feature dimension
        """
        super().__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1),
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input features [batch_size, seq_len, input_dim]
            
        Returns:
            Tuple of (pooled_features, attention_weights)
        """
        # Compute attention weights
        attention_weights = self.attention(x)  # [batch, seq_len, 1]
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Apply attention weights
        pooled = torch.sum(x * attention_weights, dim=1)  # [batch, input_dim]
        
        return pooled, attention_weights.squeeze(-1)
