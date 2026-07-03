"""Molecular Graph Transformer (MGT) drug encoder.

Applies transformer-style multi-head self-attention to molecular graphs, using
the graph topology to bias attention scores (inspired by Graphormer).  Each
atom is treated as a token; edges provide structural bias so the model can
distinguish between bonded and non-bonded atom pairs.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class GraphPositionalEncoding(nn.Module):
    """Encodes shortest-path distances between atoms as attention biases.

    Args:
        max_distance: Maximum graph distance to distinguish (longer distances
            are clipped to this value).
        num_heads: Number of attention heads (one bias learned per head).
    """

    def __init__(self, max_distance: int = 10, num_heads: int = 8) -> None:
        super().__init__()
        self.max_distance = max_distance
        # +2: index 0 = padding / unreachable, index 1..max_distance = actual dist
        self.bias_embedding = nn.Embedding(max_distance + 2, num_heads)

    def forward(self, dist_matrix: Tensor) -> Tensor:
        """Return per-head attention bias from pairwise graph distances.

        Args:
            dist_matrix: Pairwise shortest-path distances of shape
                ``(batch, n_atoms, n_atoms)`` with -1 for padding atoms.

        Returns:
            Attention bias of shape ``(batch, num_heads, n_atoms, n_atoms)``.
        """
        # Clip to [0, max_distance]; treat -1 (padding) as 0
        clipped = dist_matrix.clamp(0, self.max_distance)
        # bias_embedding: (batch, N, N, num_heads) → (batch, num_heads, N, N)
        bias = self.bias_embedding(clipped)
        return bias.permute(0, 3, 1, 2)


class MolecularGraphAttention(nn.Module):
    """Multi-head self-attention over atom tokens with graph-distance bias.

    Args:
        embed_dim: Atom embedding dimension.
        num_heads: Number of attention heads.
        max_distance: Maximum graph distance for positional bias.
        dropout: Attention dropout probability.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        max_distance: int = 10,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.graph_pe = GraphPositionalEncoding(max_distance, num_heads)

    def forward(
        self,
        x: Tensor,
        dist_matrix: Tensor,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Compute graph-biased self-attention.

        Args:
            x: Atom embeddings of shape ``(batch, n_atoms, embed_dim)``.
            dist_matrix: Pairwise distances ``(batch, n_atoms, n_atoms)``.
            key_padding_mask: Boolean mask ``(batch, n_atoms)`` where ``True``
                indicates a padding atom that should be ignored.

        Returns:
            Tuple of (output tensor same shape as *x*, attention weights
            ``(batch, num_heads, n_atoms, n_atoms)``).
        """
        B, N, _ = x.shape

        Q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # (batch, num_heads, N, N)
        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Add graph-distance positional bias
        attn = attn + self.graph_pe(dist_matrix)

        if key_padding_mask is not None:
            # Mask shape: (batch, 1, 1, N)
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.attn_drop(attn_weights)

        out = torch.matmul(attn_weights, V)  # (batch, num_heads, N, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, N, self.embed_dim)
        out = self.out_proj(out)
        return out, attn_weights


class MGTLayer(nn.Module):
    """Single Molecular Graph Transformer layer (attention + FFN).

    Args:
        embed_dim: Atom embedding dimension.
        ffn_dim: Feed-forward network hidden size.
        num_heads: Number of attention heads.
        max_distance: Maximum graph distance for positional bias.
        dropout: Dropout probability applied after attention and FFN.
    """

    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        num_heads: int = 8,
        max_distance: int = 10,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attn = MolecularGraphAttention(embed_dim, num_heads, max_distance, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: Tensor,
        dist_matrix: Tensor,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        # Pre-norm architecture (more stable for transformer pre-training)
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.attn(x, dist_matrix, key_padding_mask)
        x = residual + attn_out

        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)
        return x


class MolecularGraphTransformer(nn.Module):
    """Drug encoder based on Molecular Graph Transformer.

    Encodes a molecular graph (atoms + bond/distance information) into a
    fixed-size vector representation using stacked MGT layers followed by
    a global pooling step.

    Args:
        atom_feat_dim: Dimensionality of input atom feature vectors.
        embed_dim: Internal transformer embedding size.
        num_layers: Number of MGT layers.
        num_heads: Number of attention heads per layer.
        ffn_dim: Feed-forward network hidden size.
        max_distance: Maximum graph distance for structural bias.
        dropout: Dropout probability.
        pooling: Graph pooling strategy – ``"mean"``, ``"max"``, or
            ``"attention"`` (learned weighted pooling).
        output_dim: Size of the output drug representation vector.  If
            ``None``, defaults to *embed_dim*.
    """

    def __init__(
        self,
        atom_feat_dim: int = 78,
        embed_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 512,
        max_distance: int = 10,
        dropout: float = 0.1,
        pooling: str = "attention",
        output_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        # Project raw atom features to embed_dim
        self.atom_embedding = nn.Sequential(
            nn.Linear(atom_feat_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        self.layers = nn.ModuleList(
            [
                MGTLayer(embed_dim, ffn_dim, num_heads, max_distance, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.pooling = pooling
        if pooling == "attention":
            self.pool_attn = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.Tanh(),
                nn.Linear(embed_dim // 2, 1),
            )

        out_dim = output_dim or embed_dim
        self.output_proj = (
            nn.Linear(embed_dim, out_dim) if out_dim != embed_dim else nn.Identity()
        )

    def forward(
        self,
        atom_features: Tensor,
        dist_matrix: Tensor,
        atom_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Encode a batch of molecular graphs.

        Args:
            atom_features: Atom feature matrix of shape
                ``(batch, max_atoms, atom_feat_dim)``.
            dist_matrix: Shortest-path distance matrix of shape
                ``(batch, max_atoms, max_atoms)``.  Padding atoms should have
                distance -1 to all other atoms.
            atom_mask: Boolean padding mask of shape ``(batch, max_atoms)``
                where ``True`` marks padding atoms.

        Returns:
            Drug representation of shape ``(batch, output_dim)``.
        """
        x = self.atom_embedding(atom_features)

        for layer in self.layers:
            x = layer(x, dist_matrix, atom_mask)

        x = self.norm(x)

        # Global pooling over non-padding atoms
        if atom_mask is not None:
            # Invert mask for weighting: valid atoms = 1, padding = 0
            valid = (~atom_mask).float().unsqueeze(-1)  # (B, N, 1)
        else:
            valid = torch.ones(x.size(0), x.size(1), 1, device=x.device)

        if self.pooling == "mean":
            pooled = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1e-6)
        elif self.pooling == "max":
            x_masked = x + (1 - valid) * (-1e9)
            pooled = x_masked.max(dim=1).values
        else:  # attention
            scores = self.pool_attn(x)  # (B, N, 1)
            scores = scores + (1 - valid) * (-1e9)
            weights = F.softmax(scores, dim=1)  # (B, N, 1)
            pooled = (x * weights).sum(dim=1)  # (B, embed_dim)

        return self.output_proj(pooled)
