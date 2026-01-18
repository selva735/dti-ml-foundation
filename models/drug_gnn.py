"""
Graph Neural Network for Drug Molecular Representation.
Implements message passing on molecular graphs with attention mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool


class DrugGNN(nn.Module):
    """
    Graph Neural Network for processing drug molecular graphs.
    
    Uses Graph Attention Networks (GAT) for learning molecular representations
    with multi-head attention and residual connections.
    
    Args:
        in_features (int): Number of input node features
        hidden_dim (int): Hidden layer dimension
        out_dim (int): Output embedding dimension
        num_layers (int): Number of graph convolution layers
        heads (int): Number of attention heads in GAT
        dropout (float): Dropout rate
        pooling (str): Graph pooling method ('mean', 'max', or 'both')
    """
    
    def __init__(
        self,
        in_features=78,  # Standard molecular fingerprint size
        hidden_dim=256,
        out_dim=256,
        num_layers=3,
        heads=4,
        dropout=0.2,
        pooling='both'
    ):
        super(DrugGNN, self).__init__()
        
        self.num_layers = num_layers
        self.pooling = pooling
        self.dropout = dropout
        
        # Initial projection layer
        self.node_embed = nn.Linear(in_features, hidden_dim)
        
        # Graph attention layers
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_dim if i == 0 else hidden_dim * heads
            out_dim_layer = hidden_dim if i < num_layers - 1 else hidden_dim
            
            self.gat_layers.append(
                GATConv(
                    in_dim,
                    out_dim_layer,
                    heads=heads if i < num_layers - 1 else 1,
                    dropout=dropout,
                    concat=True if i < num_layers - 1 else False
                )
            )
        
        # Batch normalization layers
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim * heads if i < num_layers - 1 else hidden_dim)
            for i in range(num_layers)
        ])
        
        # Output projection
        pool_dim = hidden_dim * 2 if pooling == 'both' else hidden_dim
        self.fc_out = nn.Sequential(
            nn.Linear(pool_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim)
        )
        
    def forward(self, x, edge_index, batch):
        """
        Forward pass through the drug GNN.
        
        Args:
            x (torch.Tensor): Node features [num_nodes, in_features]
            edge_index (torch.Tensor): Graph connectivity [2, num_edges]
            batch (torch.Tensor): Batch assignment [num_nodes]
            
        Returns:
            torch.Tensor: Graph embeddings [batch_size, out_dim]
        """
        # Initial node embedding
        x = self.node_embed(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Graph convolution layers
        for i, (gat, bn) in enumerate(zip(self.gat_layers, self.batch_norms)):
            x_residual = x
            x = gat(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Residual connection for layers after the first
            if i > 0 and x.shape[-1] == x_residual.shape[-1]:
                x = x + x_residual
        
        # Graph pooling
        if self.pooling == 'mean':
            graph_embed = global_mean_pool(x, batch)
        elif self.pooling == 'max':
            graph_embed = global_max_pool(x, batch)
        elif self.pooling == 'both':
            mean_pool = global_mean_pool(x, batch)
            max_pool = global_max_pool(x, batch)
            graph_embed = torch.cat([mean_pool, max_pool], dim=-1)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")
        
        # Output projection
        out = self.fc_out(graph_embed)
        
        return out
    
    def get_attention_weights(self, x, edge_index):
        """
        Extract attention weights from GAT layers for interpretability.
        
        Args:
            x (torch.Tensor): Node features
            edge_index (torch.Tensor): Graph connectivity
            
        Returns:
            list: Attention weights for each layer
        """
        attention_weights = []
        
        x = self.node_embed(x)
        x = F.relu(x)
        
        for gat in self.gat_layers:
            x, (edge_idx, alpha) = gat(x, edge_index, return_attention_weights=True)
            attention_weights.append((edge_idx, alpha))
            x = F.relu(x)
            
        return attention_weights
