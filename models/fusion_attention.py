"""
Multi-Modal Fusion with Cross-Attention Mechanism.
Fuses drug, protein, and topological features using attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FusionAttention(nn.Module):
    """
    Multi-modal fusion module using cross-attention mechanisms.
    
    Implements multi-head cross-attention to fuse representations from:
    - Drug molecular graphs (GNN)
    - Protein sequences (PLM)
    - Topological features (TDA)
    
    Args:
        drug_dim (int): Drug embedding dimension
        protein_dim (int): Protein embedding dimension
        tda_dim (int): TDA embedding dimension
        hidden_dim (int): Hidden dimension for attention
        num_heads (int): Number of attention heads
        dropout (float): Dropout rate
        fusion_type (str): Type of fusion ('concat', 'attention', 'gated')
    """
    
    def __init__(
        self,
        drug_dim=256,
        protein_dim=256,
        tda_dim=256,
        hidden_dim=512,
        num_heads=8,
        dropout=0.2,
        fusion_type='attention'
    ):
        super(FusionAttention, self).__init__()
        
        self.drug_dim = drug_dim
        self.protein_dim = protein_dim
        self.tda_dim = tda_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.fusion_type = fusion_type
        
        # Project all modalities to same dimension
        self.drug_proj = nn.Linear(drug_dim, hidden_dim)
        self.protein_proj = nn.Linear(protein_dim, hidden_dim)
        self.tda_proj = nn.Linear(tda_dim, hidden_dim)
        
        # Multi-head cross-attention layers
        self.drug_protein_attn = MultiHeadCrossAttention(
            hidden_dim, num_heads, dropout
        )
        self.drug_tda_attn = MultiHeadCrossAttention(
            hidden_dim, num_heads, dropout
        )
        self.protein_tda_attn = MultiHeadCrossAttention(
            hidden_dim, num_heads, dropout
        )
        
        # Self-attention for final fusion
        self.self_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Gated fusion mechanism
        if fusion_type == 'gated':
            self.gate_drug = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid()
            )
            self.gate_protein = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid()
            )
            self.gate_tda = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid()
            )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Modal importance learning
        self.modal_importance = nn.Sequential(
            nn.Linear(hidden_dim * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, drug_embed, protein_embed, tda_embed):
        """
        Forward pass through fusion attention.
        
        Args:
            drug_embed (torch.Tensor): Drug embeddings [batch_size, drug_dim]
            protein_embed (torch.Tensor): Protein embeddings [batch_size, protein_dim]
            tda_embed (torch.Tensor): TDA embeddings [batch_size, tda_dim]
            
        Returns:
            torch.Tensor: Fused embeddings [batch_size, hidden_dim]
            dict: Attention weights for interpretability
        """
        batch_size = drug_embed.size(0)
        
        # Project to common dimension
        drug_h = self.drug_proj(drug_embed)  # [B, H]
        protein_h = self.protein_proj(protein_embed)  # [B, H]
        tda_h = self.tda_proj(tda_embed)  # [B, H]
        
        # Add sequence dimension for attention
        drug_h = drug_h.unsqueeze(1)  # [B, 1, H]
        protein_h = protein_h.unsqueeze(1)  # [B, 1, H]
        tda_h = tda_h.unsqueeze(1)  # [B, 1, H]
        
        # Cross-modal attention
        # Drug attends to Protein
        drug_protein_attn, dp_weights = self.drug_protein_attn(
            drug_h, protein_h, protein_h
        )
        drug_h_updated = self.norm1(drug_h + drug_protein_attn)
        
        # Drug attends to TDA
        drug_tda_attn, dt_weights = self.drug_tda_attn(
            drug_h_updated, tda_h, tda_h
        )
        drug_h_final = self.norm2(drug_h_updated + drug_tda_attn)
        
        # Protein attends to TDA
        protein_tda_attn, pt_weights = self.protein_tda_attn(
            protein_h, tda_h, tda_h
        )
        protein_h_final = self.norm3(protein_h + protein_tda_attn)
        
        # Combine all modalities
        combined = torch.cat([drug_h_final, protein_h_final, tda_h], dim=1)  # [B, 3, H]
        
        if self.fusion_type == 'attention':
            # Self-attention over all modalities
            fused, attn_weights = self.self_attention(combined, combined, combined)
            fused = fused.mean(dim=1)  # [B, H]
        elif self.fusion_type == 'gated':
            # Gated fusion
            drug_flat = drug_h_final.squeeze(1)
            protein_flat = protein_h_final.squeeze(1)
            tda_flat = tda_h.squeeze(1)
            
            # Compute gates
            gate_input_d = torch.cat([drug_flat, protein_flat], dim=-1)
            gate_input_p = torch.cat([protein_flat, tda_flat], dim=-1)
            gate_input_t = torch.cat([tda_flat, drug_flat], dim=-1)
            
            gate_d = self.gate_drug(gate_input_d)
            gate_p = self.gate_protein(gate_input_p)
            gate_t = self.gate_tda(gate_input_t)
            
            fused = gate_d * drug_flat + gate_p * protein_flat + gate_t * tda_flat
            attn_weights = torch.stack([gate_d, gate_p, gate_t], dim=1)
        else:  # concat
            fused = combined.mean(dim=1)
            attn_weights = None
        
        # Apply feed-forward network
        fused = fused + self.ffn(fused)
        
        # Compute modal importance
        all_modals = torch.cat([
            drug_h_final.squeeze(1),
            protein_h_final.squeeze(1),
            tda_h.squeeze(1)
        ], dim=-1)
        modal_weights = self.modal_importance(all_modals)
        
        # Weighted fusion with modal importance
        weighted_fused = (
            modal_weights[:, 0:1] * drug_h_final.squeeze(1) +
            modal_weights[:, 1:2] * protein_h_final.squeeze(1) +
            modal_weights[:, 2:3] * tda_h.squeeze(1)
        )
        
        # Combine attention-based and importance-weighted fusion
        final_fused = (fused + weighted_fused) / 2
        
        # Return attention weights for interpretability
        attention_info = {
            'drug_protein': dp_weights,
            'drug_tda': dt_weights,
            'protein_tda': pt_weights,
            'modal_importance': modal_weights
        }
        
        return final_fused, attention_info


class MultiHeadCrossAttention(nn.Module):
    """
    Multi-head cross-attention layer.
    
    Args:
        hidden_dim (int): Hidden dimension
        num_heads (int): Number of attention heads
        dropout (float): Dropout rate
    """
    
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super(MultiHeadCrossAttention, self).__init__()
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, query, key, value, mask=None):
        """
        Forward pass of cross-attention.
        
        Args:
            query (torch.Tensor): Query tensor [B, seq_len_q, hidden_dim]
            key (torch.Tensor): Key tensor [B, seq_len_k, hidden_dim]
            value (torch.Tensor): Value tensor [B, seq_len_v, hidden_dim]
            mask (torch.Tensor, optional): Attention mask
            
        Returns:
            torch.Tensor: Attention output [B, seq_len_q, hidden_dim]
            torch.Tensor: Attention weights [B, num_heads, seq_len_q, seq_len_k]
        """
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.q_linear(query)  # [B, seq_len_q, hidden_dim]
        K = self.k_linear(key)    # [B, seq_len_k, hidden_dim]
        V = self.v_linear(value)  # [B, seq_len_v, hidden_dim]
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        
        # Reshape and project output
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.hidden_dim
        )
        output = self.out_linear(context)
        
        return output, attn_weights
