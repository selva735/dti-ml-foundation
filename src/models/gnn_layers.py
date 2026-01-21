"""Graph Neural Network layers for molecular graph encoding.

This module implements GAT and GIN layers for processing molecular graphs.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GINConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class GATLayer(nn.Module):
    """Graph Attention Network layer.
    
    Uses multi-head attention to aggregate neighbor information.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 4,
        dropout: float = 0.2,
        edge_dim: Optional[int] = None,
    ):
        """Initialize GAT layer.
        
        Args:
            in_channels: Input feature dimension
            out_channels: Output feature dimension per head
            heads: Number of attention heads
            dropout: Dropout rate
            edge_dim: Edge feature dimension (optional)
        """
        super().__init__()
        
        self.conv = GATConv(
            in_channels,
            out_channels,
            heads=heads,
            dropout=dropout,
            edge_dim=edge_dim,
            concat=True,
        )
        
        self.batch_norm = nn.BatchNorm1d(out_channels * heads)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim] (optional)
            
        Returns:
            Updated node features [num_nodes, out_channels * heads]
        """
        x = self.conv(x, edge_index, edge_attr=edge_attr)
        x = self.batch_norm(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x


class GINLayer(nn.Module):
    """Graph Isomorphism Network layer.
    
    More powerful than GCN for distinguishing graph structures.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.2,
    ):
        """Initialize GIN layer.
        
        Args:
            in_channels: Input feature dimension
            out_channels: Output feature dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        # MLP for message aggregation
        mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels * 2),
            nn.BatchNorm1d(out_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels * 2, out_channels),
        )
        
        self.conv = GINConv(mlp, train_eps=True)
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            
        Returns:
            Updated node features [num_nodes, out_channels]
        """
        x = self.conv(x, edge_index)
        x = self.batch_norm(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x


class GraphEncoder(nn.Module):
    """Multi-layer graph encoder for molecular graphs.
    
    Stacks multiple GNN layers and applies graph-level pooling.
    """
    
    def __init__(
        self,
        node_feat_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        gnn_type: str = "gat",
        heads: int = 4,
        dropout: float = 0.2,
        edge_feat_dim: Optional[int] = None,
        pooling: str = "mean",
    ):
        """Initialize graph encoder.
        
        Args:
            node_feat_dim: Input node feature dimension
            hidden_dim: Hidden dimension for GNN layers
            num_layers: Number of GNN layers
            gnn_type: Type of GNN ('gat' or 'gin')
            heads: Number of attention heads (for GAT)
            dropout: Dropout rate
            edge_feat_dim: Edge feature dimension (for GAT)
            pooling: Pooling strategy ('mean', 'max', 'add')
        """
        super().__init__()
        
        self.gnn_type = gnn_type
        self.pooling = pooling
        self.num_layers = num_layers
        
        # Create GNN layers
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            in_dim = node_feat_dim if i == 0 else hidden_dim
            
            if gnn_type == "gat":
                # For GAT, we need to account for concatenated heads
                if i > 0:
                    in_dim = hidden_dim * heads
                
                layer = GATLayer(
                    in_dim,
                    hidden_dim,
                    heads=heads,
                    dropout=dropout,
                    edge_dim=edge_feat_dim if i == 0 else None,
                )
                self.layers.append(layer)
            
            elif gnn_type == "gin":
                layer = GINLayer(in_dim, hidden_dim, dropout=dropout)
                self.layers.append(layer)
            
            else:
                raise ValueError(f"Unknown GNN type: {gnn_type}")
        
        # Output dimension
        if gnn_type == "gat":
            self.output_dim = hidden_dim * heads
        else:
            self.output_dim = hidden_dim
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through graph encoder.
        
        Args:
            x: Node features [num_nodes, node_feat_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_feat_dim] (optional)
            batch: Batch assignment [num_nodes] (optional)
            
        Returns:
            Graph-level embeddings [batch_size, output_dim]
        """
        # Apply GNN layers
        for i, layer in enumerate(self.layers):
            if self.gnn_type == "gat" and i == 0 and edge_attr is not None:
                x = layer(x, edge_index, edge_attr)
            elif self.gnn_type == "gat":
                x = layer(x, edge_index)
            else:  # GIN
                x = layer(x, edge_index)
        
        # Graph-level pooling
        if batch is None:
            # Single graph
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        if self.pooling == "mean":
            graph_emb = global_mean_pool(x, batch)
        elif self.pooling == "max":
            graph_emb = global_max_pool(x, batch)
        elif self.pooling == "add":
            graph_emb = global_add_pool(x, batch)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        
        return graph_emb
