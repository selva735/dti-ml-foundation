"""
Topological Data Analysis (TDA) Encoder.
Encodes topological features like persistence diagrams and Betti curves.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TDAEncoder(nn.Module):
    """
    Topological Data Analysis encoder for capturing structural patterns.
    
    Processes topological features including:
    - Persistence diagrams (birth-death pairs)
    - Betti numbers and curves
    - Persistence landscapes
    
    Args:
        input_dim (int): Dimension of input topological features
        hidden_dim (int): Hidden layer dimension
        out_dim (int): Output embedding dimension
        num_layers (int): Number of encoding layers
        use_persistence_images (bool): Whether to use persistence images
        dropout (float): Dropout rate
    """
    
    def __init__(
        self,
        input_dim=100,
        hidden_dim=128,
        out_dim=256,
        num_layers=3,
        use_persistence_images=True,
        dropout=0.2
    ):
        super(TDAEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.use_persistence_images = use_persistence_images
        self.dropout = dropout
        
        # Persistence diagram encoder
        if use_persistence_images:
            # CNN for persistence images
            self.persistence_cnn = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(64 * (input_dim // 4) * (input_dim // 4), hidden_dim)
            )
        else:
            # MLP for vectorized persistence features
            self.persistence_mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        
        # Betti curve encoder
        self.betti_encoder = nn.Sequential(
            nn.Linear(50, hidden_dim // 2),  # Assuming 50 points in Betti curve
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Persistence landscape encoder
        self.landscape_encoder = nn.Sequential(
            nn.Linear(50, hidden_dim // 2),  # Assuming 50 points in landscape
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Combined encoder layers
        combined_dim = hidden_dim + hidden_dim // 2 + hidden_dim // 2
        layers = []
        for i in range(num_layers):
            in_dim = combined_dim if i == 0 else hidden_dim
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout)
            ])
        
        self.encoder = nn.Sequential(*layers)
        
        # Output projection
        self.fc_out = nn.Linear(hidden_dim, out_dim)
        
        # Learnable topological attention
        self.tda_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 3)  # 3 components: persistence, betti, landscape
        )
        
    def forward(self, persistence_features, betti_curves, landscapes):
        """
        Forward pass through the TDA encoder.
        
        Args:
            persistence_features (torch.Tensor): Persistence diagrams or images
                                                [batch_size, channels, H, W] or [batch_size, features]
            betti_curves (torch.Tensor): Betti number curves [batch_size, num_points]
            landscapes (torch.Tensor): Persistence landscapes [batch_size, num_points]
            
        Returns:
            torch.Tensor: TDA embeddings [batch_size, out_dim]
        """
        # Encode persistence features
        if self.use_persistence_images and persistence_features.dim() == 4:
            persistence_embed = self.persistence_cnn(persistence_features)
        else:
            if persistence_features.dim() == 4:
                persistence_features = persistence_features.view(persistence_features.size(0), -1)
            persistence_embed = self.persistence_mlp(persistence_features)
        
        # Encode Betti curves
        betti_embed = self.betti_encoder(betti_curves)
        
        # Encode persistence landscapes
        landscape_embed = self.landscape_encoder(landscapes)
        
        # Concatenate all topological features
        combined = torch.cat([persistence_embed, betti_embed, landscape_embed], dim=-1)
        
        # Apply encoder layers
        encoded = self.encoder(combined)
        
        # Apply attention to weight different topological components
        attention_weights = F.softmax(self.tda_attention(encoded), dim=-1)
        
        # Weighted combination
        weighted_persistence = persistence_embed * attention_weights[:, 0:1]
        weighted_betti = betti_embed * attention_weights[:, 1:2]
        weighted_landscape = landscape_embed * attention_weights[:, 2:3]
        
        # Combine with residual connection
        attended = torch.cat([weighted_persistence, weighted_betti, weighted_landscape], dim=-1)
        attended_encoded = self.encoder(attended)
        
        # Output projection with residual
        out = self.fc_out(encoded + attended_encoded)
        
        return out
    
    def compute_persistence_statistics(self, persistence_features):
        """
        Compute statistical features from persistence diagrams.
        
        Args:
            persistence_features (torch.Tensor): Persistence features
            
        Returns:
            dict: Dictionary of topological statistics
        """
        stats = {}
        
        if persistence_features.dim() == 4:
            # Flatten if image format
            persistence_features = persistence_features.view(persistence_features.size(0), -1)
        
        # Compute basic statistics
        stats['mean'] = torch.mean(persistence_features, dim=-1)
        stats['std'] = torch.std(persistence_features, dim=-1)
        stats['max'] = torch.max(persistence_features, dim=-1)[0]
        stats['min'] = torch.min(persistence_features, dim=-1)[0]
        
        return stats


def compute_persistence_image(diagram, resolution=20, sigma=0.1):
    """
    Convert persistence diagram to persistence image.
    
    Args:
        diagram (np.ndarray): Persistence diagram (birth-death pairs) [N, 2]
        resolution (int): Resolution of the image
        sigma (float): Gaussian kernel bandwidth
        
    Returns:
        np.ndarray: Persistence image [resolution, resolution]
    """
    # Create grid
    birth_range = np.linspace(diagram[:, 0].min(), diagram[:, 0].max(), resolution)
    death_range = np.linspace(diagram[:, 1].min(), diagram[:, 1].max(), resolution)
    
    # Initialize image
    image = np.zeros((resolution, resolution))
    
    # Weight function (persistence)
    weights = diagram[:, 1] - diagram[:, 0]
    
    # Place Gaussians at each point
    for i, (b, d) in enumerate(diagram):
        for x_idx, x_val in enumerate(birth_range):
            for y_idx, y_val in enumerate(death_range):
                dist = (b - x_val)**2 + (d - y_val)**2
                image[x_idx, y_idx] += weights[i] * np.exp(-dist / (2 * sigma**2))
    
    return image


def compute_betti_curve(diagram, num_points=50):
    """
    Compute Betti curve from persistence diagram.
    
    Args:
        diagram (np.ndarray): Persistence diagram [N, 2]
        num_points (int): Number of points in the curve
        
    Returns:
        np.ndarray: Betti curve [num_points]
    """
    filtration_values = np.linspace(diagram[:, 0].min(), diagram[:, 1].max(), num_points)
    betti_curve = np.zeros(num_points)
    
    for i, t in enumerate(filtration_values):
        # Count features alive at filtration value t
        betti_curve[i] = np.sum((diagram[:, 0] <= t) & (diagram[:, 1] > t))
    
    return betti_curve
