"""Bidirectional Cross-Modal Attention Fusion.

Implements a symmetric cross-attention mechanism where:
  - Drug representations attend to protein context.
  - Protein representations attend to drug context.

The attended outputs are then fused and projected to produce the final
combined drug–protein interaction representation.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CrossAttentionBlock(nn.Module):
    """Single cross-attention block: query from one modality, key/value from another.

    Args:
        query_dim: Dimensionality of the query modality.
        key_dim: Dimensionality of the key/value modality.
        embed_dim: Internal embedding dimension.
        num_heads: Number of attention heads.
        dropout: Dropout on attention weights.
    """

    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(query_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(key_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(key_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.norm_q = nn.LayerNorm(query_dim)
        self.norm_k = nn.LayerNorm(key_dim)

    def forward(
        self,
        query: Tensor,
        key_value: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Perform cross-attention.

        Args:
            query: Query tensor of shape ``(batch, query_dim)`` (single token
                per sample).
            key_value: Key/Value tensor of shape ``(batch, key_dim)`` (also
                treated as a single token here – both modalities produce global
                pooled representations).

        Returns:
            Tuple of (output of shape ``(batch, embed_dim)``,
            attention weights ``(batch, num_heads, 1, 1)``).
        """
        # Unsqueeze to add sequence dimension for matmul compatibility
        q = self.q_proj(self.norm_q(query)).unsqueeze(1)   # (B, 1, E)
        k = self.k_proj(self.norm_k(key_value)).unsqueeze(1)  # (B, 1, E)
        v = self.v_proj(key_value).unsqueeze(1)              # (B, 1, E)

        B, _, E = q.shape
        q = q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.attn_drop(attn_weights)

        out = torch.matmul(attn_weights, v)  # (B, num_heads, 1, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, E)
        out = self.out_proj(out)
        return out, attn_weights


class BidirectionalCrossAttention(nn.Module):
    """Bidirectional cross-modal attention between drug and protein representations.

    Computes:
      1. Drug-attends-to-protein: drug representation enriched with protein context.
      2. Protein-attends-to-drug: protein representation enriched with drug context.

    The two enriched representations are concatenated and projected to produce
    the final interaction embedding.

    Args:
        drug_dim: Dimensionality of drug encoder output.
        protein_dim: Dimensionality of protein encoder output.
        cross_dim: Internal cross-attention dimension.
        num_heads: Number of attention heads.
        num_layers: Number of stacked cross-attention layers.
        dropout: Dropout probability.
        output_dim: Size of the final fused representation.
    """

    def __init__(
        self,
        drug_dim: int = 256,
        protein_dim: int = 256,
        cross_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        output_dim: int = 512,
    ) -> None:
        super().__init__()
        self.drug_dim = drug_dim
        self.protein_dim = protein_dim
        self.cross_dim = cross_dim

        # Stack of bidirectional cross-attention layers
        self.drug_to_protein_layers = nn.ModuleList(
            [
                CrossAttentionBlock(
                    query_dim=cross_dim if i > 0 else drug_dim,
                    key_dim=cross_dim if i > 0 else protein_dim,
                    embed_dim=cross_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for i in range(num_layers)
            ]
        )
        self.protein_to_drug_layers = nn.ModuleList(
            [
                CrossAttentionBlock(
                    query_dim=cross_dim if i > 0 else protein_dim,
                    key_dim=cross_dim if i > 0 else drug_dim,
                    embed_dim=cross_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for i in range(num_layers)
            ]
        )

        # Feed-forward refinement after cross-attention
        self.drug_ffn = nn.Sequential(
            nn.LayerNorm(cross_dim),
            nn.Linear(cross_dim, cross_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(cross_dim * 2, cross_dim),
        )
        self.protein_ffn = nn.Sequential(
            nn.LayerNorm(cross_dim),
            nn.Linear(cross_dim, cross_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(cross_dim * 2, cross_dim),
        )

        # Final fusion projection
        self.fusion_proj = nn.Sequential(
            nn.Linear(cross_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(
        self,
        drug_repr: Tensor,
        protein_repr: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Fuse drug and protein representations via bidirectional cross-attention.

        Args:
            drug_repr: Drug encoder output of shape ``(batch, drug_dim)``.
            protein_repr: Protein encoder output of shape ``(batch, protein_dim)``.

        Returns:
            Tuple of:
              - fused representation of shape ``(batch, output_dim)``
              - refined drug repr ``(batch, cross_dim)``
              - refined protein repr ``(batch, cross_dim)``
        """
        d = drug_repr
        p = protein_repr

        for d2p, p2d in zip(self.drug_to_protein_layers, self.protein_to_drug_layers):
            d_new, _ = d2p(d, p)
            p_new, _ = p2d(p, d)
            d = d_new
            p = p_new

        d = d + self.drug_ffn(d)
        p = p + self.protein_ffn(p)

        fused = torch.cat([d, p], dim=-1)
        return self.fusion_proj(fused), d, p
