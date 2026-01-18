"""
Evidential Deep Learning Head for Uncertainty Estimation.
Implements evidential regression for epistemic and aleatoric uncertainty.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class EvidentialHead(nn.Module):
    """
    Evidential deep learning head for uncertainty-aware predictions.
    
    Outputs four evidential parameters (gamma, nu, alpha, beta) that
    parameterize a Normal-Inverse-Gamma distribution, enabling:
    - Point predictions
    - Epistemic uncertainty (model uncertainty)
    - Aleatoric uncertainty (data uncertainty)
    - Total uncertainty
    
    Based on: "Deep Evidential Regression" (Amini et al., NeurIPS 2020)
    
    Args:
        in_dim (int): Input feature dimension
        hidden_dim (int): Hidden layer dimension
        dropout (float): Dropout rate
        use_residual (bool): Whether to use residual connections
    """
    
    def __init__(
        self,
        in_dim=512,
        hidden_dim=256,
        dropout=0.2,
        use_residual=True
    ):
        super(EvidentialHead, self).__init__()
        
        self.in_dim = in_dim
        self.use_residual = use_residual
        
        # Shared feature extraction
        self.shared_layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Evidential parameter heads
        # gamma: predicted mean
        self.gamma_head = nn.Linear(hidden_dim, 1)
        
        # nu: degrees of freedom (> 1)
        self.nu_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )
        
        # alpha: inverse gamma shape (> 1)
        self.alpha_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )
        
        # beta: inverse gamma scale (> 0)
        self.beta_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )
        
        # Alternative: classification head for binary DTI prediction
        self.binary_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)  # Binary classification: binding / non-binding
        )
        
    def forward(self, x, return_all=True):
        """
        Forward pass through evidential head.
        
        Args:
            x (torch.Tensor): Input features [batch_size, in_dim]
            return_all (bool): Whether to return all evidential parameters
            
        Returns:
            If return_all:
                dict: Dictionary containing:
                    - gamma: predicted mean
                    - nu: degrees of freedom
                    - alpha: inverse gamma shape
                    - beta: inverse gamma scale
                    - pred: point prediction
                    - epistemic_unc: epistemic uncertainty
                    - aleatoric_unc: aleatoric uncertainty
                    - total_unc: total uncertainty
            Else:
                torch.Tensor: Point predictions [batch_size, 1]
        """
        # Shared feature extraction
        features = self.shared_layers(x)
        
        # Predict evidential parameters
        gamma = self.gamma_head(features)  # Predicted mean
        nu = self.nu_head(features) + 1.0  # Ensure nu > 1
        alpha = self.alpha_head(features) + 1.0  # Ensure alpha > 1
        beta = self.beta_head(features)  # Ensure beta > 0
        
        if not return_all:
            return gamma
        
        # Compute uncertainties
        epistemic_unc = beta / (alpha - 1)  # Epistemic uncertainty
        aleatoric_unc = beta / (nu * (alpha - 1))  # Aleatoric uncertainty
        total_unc = epistemic_unc + aleatoric_unc  # Total uncertainty
        
        return {
            'gamma': gamma,
            'nu': nu,
            'alpha': alpha,
            'beta': beta,
            'pred': gamma,
            'epistemic_unc': epistemic_unc,
            'aleatoric_unc': aleatoric_unc,
            'total_unc': total_unc
        }
    
    def binary_prediction(self, x):
        """
        Make binary classification prediction.
        
        Args:
            x (torch.Tensor): Input features [batch_size, in_dim]
            
        Returns:
            torch.Tensor: Class logits [batch_size, 2]
        """
        features = self.shared_layers(x)
        return self.binary_head(features)


def evidential_loss(gamma, nu, alpha, beta, target, coeff=1.0):
    """
    Compute evidential regression loss.
    
    Args:
        gamma (torch.Tensor): Predicted mean
        nu (torch.Tensor): Degrees of freedom
        alpha (torch.Tensor): Inverse gamma shape
        beta (torch.Tensor): Inverse gamma scale
        target (torch.Tensor): Ground truth values
        coeff (float): Regularization coefficient
        
    Returns:
        torch.Tensor: Evidential loss
    """
    # Negative log likelihood
    error = (target - gamma).abs()
    
    nll = (
        0.5 * torch.log(math.pi / nu)
        - alpha * torch.log(2 * beta)
        + (alpha + 0.5) * torch.log(nu * error**2 + 2 * beta)
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
    )
    
    # Regularization term
    # Penalize over-confident predictions when wrong
    reg = error * (2 * alpha + nu)
    
    loss = nll.mean() + coeff * reg.mean()
    
    return loss


def evidential_mse_loss(gamma, nu, alpha, beta, target, coeff=1.0):
    """
    Compute MSE-based evidential loss (alternative formulation).
    
    Args:
        gamma (torch.Tensor): Predicted mean
        nu (torch.Tensor): Degrees of freedom
        alpha (torch.Tensor): Inverse gamma shape
        beta (torch.Tensor): Inverse gamma scale
        target (torch.Tensor): Ground truth values
        coeff (float): Regularization coefficient
        
    Returns:
        torch.Tensor: Evidential MSE loss
    """
    # MSE term
    mse = F.mse_loss(gamma, target, reduction='none')
    
    # Uncertainty term
    unc = beta / (alpha - 1)
    
    # Combined loss
    loss = mse * torch.exp(-unc) + unc
    
    # Regularization
    error = torch.abs(target - gamma)
    reg = error * (2 * nu + alpha)
    
    total_loss = loss.mean() + coeff * reg.mean()
    
    return total_loss


class EvidentialBinaryHead(nn.Module):
    """
    Evidential classification head for binary DTI prediction.
    
    Args:
        in_dim (int): Input feature dimension
        hidden_dim (int): Hidden layer dimension
        dropout (float): Dropout rate
    """
    
    def __init__(
        self,
        in_dim=512,
        hidden_dim=256,
        dropout=0.2
    ):
        super(EvidentialBinaryHead, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim)
        )
        
        # Evidence for each class
        self.evidence_head = nn.Sequential(
            nn.Linear(hidden_dim, 2),
            nn.Softplus()
        )
        
    def forward(self, x, return_uncertainty=True):
        """
        Forward pass with evidential classification.
        
        Args:
            x (torch.Tensor): Input features
            return_uncertainty (bool): Whether to return uncertainty
            
        Returns:
            dict: Predictions, probabilities, and uncertainties
        """
        features = self.feature_extractor(x)
        evidence = self.evidence_head(features)
        
        # Dirichlet parameters (alpha)
        alpha = evidence + 1.0
        
        # Belief mass and uncertainty
        S = alpha.sum(dim=-1, keepdim=True)
        prob = alpha / S
        uncertainty = 2.0 / S
        
        if return_uncertainty:
            return {
                'prob': prob,
                'uncertainty': uncertainty,
                'alpha': alpha,
                'evidence': evidence
            }
        else:
            return prob


def evidential_classification_loss(alpha, target, lambda_reg=0.01):
    """
    Compute evidential classification loss.
    
    Args:
        alpha (torch.Tensor): Dirichlet parameters [batch_size, num_classes]
        target (torch.Tensor): Ground truth labels [batch_size]
        lambda_reg (float): Regularization coefficient
        
    Returns:
        torch.Tensor: Evidential classification loss
    """
    S = alpha.sum(dim=-1)
    
    # Convert target to one-hot
    num_classes = alpha.size(-1)
    target_one_hot = F.one_hot(target, num_classes).float()
    
    # Expected log likelihood
    digamma_sum = torch.digamma(S)
    digamma_alpha = torch.digamma(alpha)
    log_likelihood = torch.sum(target_one_hot * (digamma_alpha - digamma_sum.unsqueeze(-1)), dim=-1)
    
    # KL divergence regularization
    kl_alpha = (alpha - 1) * (1 - target_one_hot) + 1
    kl_term = torch.lgamma(kl_alpha.sum(dim=-1)) - torch.lgamma(S)
    kl_term += torch.sum(
        torch.lgamma(alpha) - torch.lgamma(kl_alpha) + (kl_alpha - alpha) * (torch.digamma(kl_alpha) - digamma_alpha),
        dim=-1
    )
    
    loss = -log_likelihood.mean() + lambda_reg * kl_term.mean()
    
    return loss
