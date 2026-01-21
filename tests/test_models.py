"""Tests for model modules."""

import pytest
import sys
from pathlib import Path
import torch
from torch_geometric.data import Data, Batch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.gnn_layers import GATLayer, GINLayer, GraphEncoder
from models.attention import MultiHeadAttention, CrossModalAttention
from models.dti_model import DTIModel, create_dti_model


class TestGNNLayers:
    """Tests for GNN layers."""
    
    def test_gat_layer_forward(self):
        """Test GAT layer forward pass."""
        batch_size = 4
        num_nodes = 10
        in_channels = 16
        out_channels = 32
        heads = 4
        
        layer = GATLayer(in_channels, out_channels, heads=heads)
        
        x = torch.randn(num_nodes, in_channels)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        
        out = layer(x, edge_index)
        
        assert out.size() == (num_nodes, out_channels * heads)
    
    def test_gin_layer_forward(self):
        """Test GIN layer forward pass."""
        num_nodes = 10
        in_channels = 16
        out_channels = 32
        
        layer = GINLayer(in_channels, out_channels)
        
        x = torch.randn(num_nodes, in_channels)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        
        out = layer(x, edge_index)
        
        assert out.size() == (num_nodes, out_channels)
    
    def test_graph_encoder(self):
        """Test graph encoder with pooling."""
        batch_size = 2
        num_nodes = 20
        node_feat_dim = 64
        hidden_dim = 128
        
        encoder = GraphEncoder(
            node_feat_dim=node_feat_dim,
            hidden_dim=hidden_dim,
            num_layers=3,
            gnn_type='gat',
        )
        
        x = torch.randn(num_nodes, node_feat_dim)
        edge_index = torch.randint(0, num_nodes, (2, 40))
        batch = torch.cat([torch.zeros(10, dtype=torch.long), torch.ones(10, dtype=torch.long)])
        
        out = encoder(x, edge_index, batch=batch)
        
        assert out.size(0) == batch_size
        assert out.size(1) == encoder.output_dim


class TestAttention:
    """Tests for attention mechanisms."""
    
    def test_multi_head_attention(self):
        """Test multi-head attention."""
        batch_size = 4
        seq_len = 10
        embed_dim = 128
        num_heads = 8
        
        attention = MultiHeadAttention(embed_dim, num_heads)
        
        query = torch.randn(batch_size, seq_len, embed_dim)
        key = torch.randn(batch_size, seq_len, embed_dim)
        value = torch.randn(batch_size, seq_len, embed_dim)
        
        out, _ = attention(query, key, value)
        
        assert out.size() == (batch_size, seq_len, embed_dim)
    
    def test_cross_modal_attention(self):
        """Test cross-modal attention."""
        batch_size = 4
        drug_dim = 256
        protein_dim = 480
        hidden_dim = 256
        
        attention = CrossModalAttention(
            drug_dim, protein_dim, hidden_dim
        )
        
        drug_emb = torch.randn(batch_size, drug_dim)
        protein_emb = torch.randn(batch_size, protein_dim)
        
        drug_context, protein_context, _ = attention(drug_emb, protein_emb)
        
        assert drug_context.size() == (batch_size, hidden_dim)
        assert protein_context.size() == (batch_size, hidden_dim)


class TestDTIModel:
    """Tests for DTI model."""
    
    def test_model_creation(self):
        """Test model creation."""
        feature_dims = {
            'node_feat_dim': 64,
            'edge_feat_dim': 7,
            'protein_emb_dim': 480,
        }
        
        model_config = {
            'hidden_dim': 128,
            'gnn_layers': 2,
            'dropout': 0.2,
        }
        
        model = create_dti_model(feature_dims, model_config)
        
        assert isinstance(model, DTIModel)
        
        # Count parameters
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0
    
    def test_model_forward(self):
        """Test model forward pass."""
        batch_size = 2
        num_nodes = 20
        node_feat_dim = 64
        edge_feat_dim = 7
        protein_emb_dim = 480
        
        # Create model
        feature_dims = {
            'node_feat_dim': node_feat_dim,
            'edge_feat_dim': edge_feat_dim,
            'protein_emb_dim': protein_emb_dim,
        }
        
        model = create_dti_model(feature_dims, {'hidden_dim': 128, 'gnn_layers': 2})
        
        # Create sample data
        # Drug graph
        x = torch.randn(num_nodes, node_feat_dim)
        edge_index = torch.randint(0, num_nodes, (2, 40))
        edge_attr = torch.randn(40, edge_feat_dim)
        batch_vec = torch.cat([torch.zeros(10, dtype=torch.long), torch.ones(10, dtype=torch.long)])
        
        drug_graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        drug_graph.batch = batch_vec
        
        # Protein embedding
        protein_emb = torch.randn(batch_size, protein_emb_dim)
        
        # Forward pass
        predictions, _ = model(drug_graph, protein_emb)
        
        assert predictions.size() == (batch_size, 1)
    
    def test_model_with_attention_weights(self):
        """Test getting attention weights."""
        batch_size = 2
        num_nodes = 20
        node_feat_dim = 64
        protein_emb_dim = 480
        
        # Create model
        feature_dims = {
            'node_feat_dim': node_feat_dim,
            'edge_feat_dim': 7,
            'protein_emb_dim': protein_emb_dim,
        }
        
        model = create_dti_model(feature_dims, {'hidden_dim': 128})
        
        # Create sample data
        x = torch.randn(num_nodes, node_feat_dim)
        edge_index = torch.randint(0, num_nodes, (2, 40))
        batch_vec = torch.cat([torch.zeros(10, dtype=torch.long), torch.ones(10, dtype=torch.long)])
        
        drug_graph = Data(x=x, edge_index=edge_index)
        drug_graph.batch = batch_vec
        
        protein_emb = torch.randn(batch_size, protein_emb_dim)
        
        # Forward pass with attention weights
        predictions, attention_weights = model(drug_graph, protein_emb, return_attention=True)
        
        assert predictions.size() == (batch_size, 1)
        assert attention_weights is not None
