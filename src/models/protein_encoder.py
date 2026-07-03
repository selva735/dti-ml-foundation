"""Protein Sequence Transformer encoder.

Encodes amino-acid sequences using a lightweight transformer with sinusoidal
positional encoding.  When pre-trained PLM embeddings are provided the module
can optionally project them instead of encoding from tokens, allowing the
model to leverage large-scale protein language model knowledge.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# Standard 20 amino acids + special tokens (X=unknown, padding)
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWYX"
AA_TO_IDX: dict[str, int] = {aa: i + 1 for i, aa in enumerate(AMINO_ACIDS)}
AA_VOCAB_SIZE = len(AMINO_ACIDS) + 2  # +1 for padding, +1 for CLS token
CLS_IDX = len(AMINO_ACIDS) + 1
PAD_IDX = 0


def tokenize_sequence(seq: str, max_len: int = 1000) -> list[int]:
    """Convert an amino-acid string to a list of integer token IDs.

    A CLS token is prepended; the sequence is truncated (not padded) to
    *max_len* residues (excluding CLS).

    Args:
        seq: Protein sequence string (single-letter codes).
        max_len: Maximum number of residues to encode.

    Returns:
        List of integer token IDs starting with CLS_IDX.
    """
    seq = seq[:max_len].upper()
    tokens = [CLS_IDX] + [AA_TO_IDX.get(aa, AA_TO_IDX["X"]) for aa in seq]
    return tokens


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding.

    Args:
        embed_dim: Embedding dimension (must be even).
        max_len: Maximum sequence length supported.
        dropout: Dropout applied after adding positional encoding.
    """

    def __init__(self, embed_dim: int, max_len: int = 1002, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Register as buffer so it moves to the right device automatically
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """Add positional encoding to *x*.

        Args:
            x: Input tensor of shape ``(batch, seq_len, embed_dim)``.

        Returns:
            Tensor with positional encoding added, same shape as *x*.
        """
        x = x + self.pe[: x.size(1)]  # type: ignore[index]
        return self.dropout(x)


class ProteinSequenceTransformer(nn.Module):
    """Protein encoder using a transformer over amino-acid tokens.

    A [CLS] token is prepended to each sequence; its final hidden state is
    used as the sequence-level representation.  Optionally, pre-computed PLM
    embeddings can be projected directly, bypassing the token embedding stage.

    Args:
        vocab_size: Token vocabulary size (default covers standard amino
            acids + CLS + PAD).
        embed_dim: Token embedding and transformer hidden size.
        num_layers: Number of transformer encoder layers.
        num_heads: Number of attention heads.
        ffn_dim: Feed-forward network hidden size.
        max_seq_len: Maximum supported sequence length (including CLS token).
        dropout: Dropout probability.
        output_dim: Output projection size.  Defaults to *embed_dim*.
        plm_input_dim: If provided, adds a projection from PLM embedding
            space to *embed_dim* so the encoder can also accept pre-computed
            PLM embeddings instead of raw token IDs.
    """

    def __init__(
        self,
        vocab_size: int = AA_VOCAB_SIZE,
        embed_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 512,
        max_seq_len: int = 1002,
        dropout: float = 0.1,
        output_dim: Optional[int] = None,
        plm_input_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.pos_encoding = SinusoidalPositionalEncoding(embed_dim, max_seq_len, dropout)

        # Optional PLM projection
        self.plm_proj: Optional[nn.Module] = None
        if plm_input_dim is not None:
            self.plm_proj = nn.Sequential(
                nn.Linear(plm_input_dim, embed_dim),
                nn.LayerNorm(embed_dim),
            )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(embed_dim)

        out_dim = output_dim or embed_dim
        self.output_proj = (
            nn.Linear(embed_dim, out_dim) if out_dim != embed_dim else nn.Identity()
        )

    def forward(
        self,
        tokens: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        plm_embeddings: Optional[Tensor] = None,
    ) -> Tensor:
        """Encode a batch of protein sequences.

        Exactly one of *tokens* or *plm_embeddings* must be provided.

        Args:
            tokens: Integer token IDs of shape ``(batch, seq_len)`` (CLS
                token prepended).  Required when *plm_embeddings* is None.
            padding_mask: Boolean mask of shape ``(batch, seq_len)`` where
                ``True`` marks padding positions.
            plm_embeddings: Pre-computed PLM sequence embeddings of shape
                ``(batch, seq_len, plm_input_dim)`` (CLS position included).

        Returns:
            Protein representation of shape ``(batch, output_dim)`` taken
            from the CLS token position.
        """
        if plm_embeddings is not None:
            if self.plm_proj is None:
                raise ValueError(
                    "plm_input_dim must be set in __init__ to use PLM embeddings."
                )
            x = self.plm_proj(plm_embeddings)
            x = self.pos_encoding(x)
        elif tokens is not None:
            x = self.token_embedding(tokens)  # (B, L, embed_dim)
            x = self.pos_encoding(x)
        else:
            raise ValueError("Either `tokens` or `plm_embeddings` must be provided.")

        # TransformerEncoder key_padding_mask convention: True = ignore
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        x = self.norm(x)

        # CLS token is always at position 0
        cls_repr = x[:, 0, :]
        return self.output_proj(cls_repr)
