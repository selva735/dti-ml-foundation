"""
Protein Language Model (PLM) Encoder.
Uses pre-trained protein language models for protein representation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProteinPLM(nn.Module):
    """
    Protein Language Model encoder for protein sequence representation.
    
    Supports integration with pre-trained models like ESM, ProtBERT, or ProtT5.
    Includes optional fine-tuning and projection layers.
    
    Args:
        plm_model (str): Pre-trained model identifier ('esm2', 'protbert', 'prot_t5')
        embedding_dim (int): PLM embedding dimension (e.g., 1280 for ESM2)
        out_dim (int): Output embedding dimension
        freeze_plm (bool): Whether to freeze the PLM weights
        use_pooling (str): Pooling strategy ('cls', 'mean', 'max', 'both')
        dropout (float): Dropout rate
    """
    
    def __init__(
        self,
        plm_model='esm2',
        embedding_dim=1280,
        out_dim=256,
        freeze_plm=True,
        use_pooling='mean',
        dropout=0.2
    ):
        super(ProteinPLM, self).__init__()
        
        self.plm_model = plm_model
        self.embedding_dim = embedding_dim
        self.use_pooling = use_pooling
        self.freeze_plm = freeze_plm
        
        # Load pre-trained protein language model
        # Note: In practice, you would load actual pre-trained weights here
        # For this framework, we provide the structure that can be extended
        self.plm = self._load_plm(plm_model, embedding_dim)
        
        if freeze_plm:
            for param in self.plm.parameters():
                param.requires_grad = False
        
        # Projection layers
        pool_multiplier = 2 if use_pooling == 'both' else 1
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim * pool_multiplier, out_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(out_dim * 2),
            nn.Linear(out_dim * 2, out_dim)
        )
        
        # Attention pooling (learned pooling weights)
        self.attention_pooling = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
    def _load_plm(self, plm_model, embedding_dim):
        """
        Load pre-trained protein language model.
        
        In production, this would load models like:
        - ESM-2: facebook/esm2_t33_650M_UR50D
        - ProtBERT: Rostlab/prot_bert
        - ProtT5: Rostlab/prot_t5_xl_uniref50
        
        For this framework, we provide a placeholder that can be replaced.
        """
        if plm_model == 'esm2':
            # Placeholder for ESM2 model
            # In practice: import esm; model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            return nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=embedding_dim,
                    nhead=8,
                    dim_feedforward=embedding_dim * 4,
                    dropout=0.1,
                    batch_first=True
                ),
                num_layers=6
            )
        elif plm_model == 'protbert':
            # Placeholder for ProtBERT
            # In practice: from transformers import BertModel; BertModel.from_pretrained('Rostlab/prot_bert')
            return nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=embedding_dim,
                    nhead=8,
                    dim_feedforward=embedding_dim * 4,
                    dropout=0.1,
                    batch_first=True
                ),
                num_layers=6
            )
        else:
            raise ValueError(f"Unknown PLM model: {plm_model}")
    
    def forward(self, x, mask=None):
        """
        Forward pass through the protein PLM.
        
        Args:
            x (torch.Tensor): Protein embeddings or token IDs [batch_size, seq_len, embedding_dim]
                             or [batch_size, seq_len] for token IDs
            mask (torch.Tensor, optional): Attention mask [batch_size, seq_len]
            
        Returns:
            torch.Tensor: Protein embeddings [batch_size, out_dim]
        """
        # Get PLM embeddings
        if x.dim() == 2:
            # If input is token IDs, create embeddings
            x = F.embedding(x, torch.randn(25, self.embedding_dim).to(x.device))
        
        # Pass through PLM
        if self.freeze_plm:
            with torch.no_grad():
                plm_output = self.plm(x)
        else:
            plm_output = self.plm(x)
        
        # Apply pooling strategy
        if self.use_pooling == 'cls':
            # Use first token (CLS token)
            pooled = plm_output[:, 0, :]
        elif self.use_pooling == 'mean':
            # Mean pooling over sequence
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).expand(plm_output.size())
                sum_embeddings = torch.sum(plm_output * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                pooled = sum_embeddings / sum_mask
            else:
                pooled = torch.mean(plm_output, dim=1)
        elif self.use_pooling == 'max':
            # Max pooling over sequence
            pooled = torch.max(plm_output, dim=1)[0]
        elif self.use_pooling == 'both':
            # Concatenate mean and max pooling
            mean_pooled = torch.mean(plm_output, dim=1)
            max_pooled = torch.max(plm_output, dim=1)[0]
            pooled = torch.cat([mean_pooled, max_pooled], dim=-1)
        elif self.use_pooling == 'attention':
            # Attention-weighted pooling
            attention_weights = self.attention_pooling(plm_output)
            attention_weights = F.softmax(attention_weights, dim=1)
            pooled = torch.sum(plm_output * attention_weights, dim=1)
        else:
            raise ValueError(f"Unknown pooling strategy: {self.use_pooling}")
        
        # Project to output dimension
        out = self.projection(pooled)
        
        return out
    
    def get_residue_embeddings(self, x, mask=None):
        """
        Get per-residue embeddings for interpretability.
        
        Args:
            x (torch.Tensor): Protein sequence input
            mask (torch.Tensor, optional): Attention mask
            
        Returns:
            torch.Tensor: Per-residue embeddings [batch_size, seq_len, embedding_dim]
        """
        if x.dim() == 2:
            x = F.embedding(x, torch.randn(25, self.embedding_dim).to(x.device))
        
        with torch.no_grad():
            residue_embeddings = self.plm(x)
        
        return residue_embeddings
