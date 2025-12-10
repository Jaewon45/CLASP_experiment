"""
Utilities for latent space manipulations and interval computations.
"""

import torch
import numpy as np
from copy import deepcopy


def apply_gaussian_noise_to_dim(latent_code, dim_idx, noise_std=0.1):
    """
    Apply Gaussian noise to a specific latent dimension.
    
    Args:
        latent_code: torch.Tensor or np.ndarray of shape (D,) or (B, D)
        dim_idx: Index of dimension to perturb
        noise_std: Standard deviation of Gaussian noise
    
    Returns:
        Perturbed latent code
    """
    if isinstance(latent_code, np.ndarray):
        latent_code = torch.from_numpy(latent_code).float()
    
    perturbed = latent_code.clone() if isinstance(latent_code, torch.Tensor) else deepcopy(latent_code)
    
    if perturbed.dim() == 1:
        perturbed[dim_idx] += torch.randn(1).item() * noise_std
    else:
        perturbed[:, dim_idx] += torch.randn(perturbed.shape[0]).to(perturbed.device) * noise_std
    
    return perturbed


def zero_out_dim(latent_code, dim_idx):
    """
    Zero out a specific latent dimension.
    
    Args:
        latent_code: torch.Tensor or np.ndarray of shape (D,) or (B, D)
        dim_idx: Index of dimension to zero out
    
    Returns:
        Modified latent code
    """
    if isinstance(latent_code, np.ndarray):
        latent_code = torch.from_numpy(latent_code).float()
    
    zeroed = latent_code.clone() if isinstance(latent_code, torch.Tensor) else deepcopy(latent_code)
    
    if zeroed.dim() == 1:
        zeroed[dim_idx] = 0.0
    else:
        zeroed[:, dim_idx] = 0.0
    
    return zeroed


def random_resample_dim(latent_code, dim_idx, prior_mean=0.0, prior_std=1.0):
    """
    Randomly re-sample a specific latent dimension from prior distribution.
    
    Args:
        latent_code: torch.Tensor or np.ndarray of shape (D,) or (B, D)
        dim_idx: Index of dimension to re-sample
        prior_mean: Mean of prior distribution
        prior_std: Standard deviation of prior distribution
    
    Returns:
        Modified latent code
    """
    if isinstance(latent_code, np.ndarray):
        latent_code = torch.from_numpy(latent_code).float()
    
    resampled = latent_code.clone() if isinstance(latent_code, torch.Tensor) else deepcopy(latent_code)
    
    if resampled.dim() == 1:
        resampled[dim_idx] = torch.randn(1).item() * prior_std + prior_mean
    else:
        resampled[:, dim_idx] = torch.randn(resampled.shape[0]).to(resampled.device) * prior_std + prior_mean
    
    return resampled


def compute_interval_width(lower_bounds, upper_bounds):
    """
    Compute interval width: W_d = E[q_d^high - q_d^low]
    
    Args:
        lower_bounds: torch.Tensor or np.ndarray of shape (N, D) or (D,)
        upper_bounds: torch.Tensor or np.ndarray of shape (N, D) or (D,)
    
    Returns:
        Average interval width per dimension: shape (D,)
    """
    if isinstance(lower_bounds, np.ndarray):
        lower_bounds = torch.from_numpy(lower_bounds).float()
    if isinstance(upper_bounds, np.ndarray):
        upper_bounds = torch.from_numpy(upper_bounds).float()
    
    widths = upper_bounds - lower_bounds
    
    if widths.dim() == 1:
        return widths
    else:
        return widths.mean(dim=0)  # Average over samples

