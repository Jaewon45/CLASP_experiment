"""
Example usage of the extension experiments framework.
This script demonstrates how to use individual components.
"""

import os
import sys
import torch

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

from corruptions.corruption_utils import (
    apply_blur,
    apply_downsampling,
    apply_masking,
    apply_adversarial_noise,
    apply_block_dropout,
    apply_corruption_level
)
from utils.latent_utils import (
    apply_gaussian_noise_to_dim,
    zero_out_dim,
    random_resample_dim,
    compute_interval_width
)
from utils.lpips_utils import LPIPSMetric


def example_corruptions():
    """Example: Apply various corruptions to an image."""
    print("="*50)
    print("Example: Applying Corruptions")
    print("="*50)
    
    # Create a dummy image tensor (normalized to [-1, 1])
    dummy_image = torch.randn(3, 256, 256) * 0.5  # Random image
    
    # Apply different corruptions
    print("\n1. Applying blur...")
    blurred = apply_blur(dummy_image, kernel_size=5, sigma=1.0)
    print(f"   Blurred image shape: {blurred.shape}")
    
    print("\n2. Applying 8x downsampling...")
    downsampled = apply_downsampling(dummy_image, factor=8)
    print(f"   Downsampled image shape: {downsampled.shape}")
    
    print("\n3. Applying 50% masking...")
    masked, mask = apply_masking(dummy_image, mask_ratio=0.5)
    print(f"   Masked image shape: {masked.shape}, mask shape: {mask.shape}")
    
    print("\n4. Applying adversarial noise...")
    adversarial = apply_adversarial_noise(dummy_image, epsilon=0.03)
    print(f"   Adversarial image shape: {adversarial.shape}")
    
    print("\n5. Applying block dropout...")
    block_dropped = apply_block_dropout(dummy_image, block_size=32, dropout_prob=0.3)
    print(f"   Block dropout image shape: {block_dropped.shape}")
    
    print("\n6. Applying corruption level 'moderate'...")
    corrupted = apply_corruption_level(dummy_image, level='moderate')
    print(f"   Corrupted image shape: {corrupted.shape}")


def example_latent_manipulations():
    """Example: Manipulate latent codes."""
    print("\n" + "="*50)
    print("Example: Latent Code Manipulations")
    print("="*50)
    
    # Create a dummy latent code
    latent_code = torch.randn(512)  # StyleGAN latent dimension
    
    print(f"\nOriginal latent code shape: {latent_code.shape}")
    print(f"Original value at dim 0: {latent_code[0]:.4f}")
    
    # Apply Gaussian noise
    print("\n1. Applying Gaussian noise to dimension 0...")
    noisy = apply_gaussian_noise_to_dim(latent_code, dim_idx=0, noise_std=0.1)
    print(f"   Noisy value at dim 0: {noisy[0]:.4f}")
    
    # Zero out dimension
    print("\n2. Zeroing out dimension 0...")
    zeroed = zero_out_dim(latent_code, dim_idx=0)
    print(f"   Zeroed value at dim 0: {zeroed[0]:.4f}")
    
    # Random resample
    print("\n3. Randomly resampling dimension 0...")
    resampled = random_resample_dim(latent_code, dim_idx=0, prior_mean=0.0, prior_std=1.0)
    print(f"   Resampled value at dim 0: {resampled[0]:.4f}")


def example_interval_computation():
    """Example: Compute interval widths."""
    print("\n" + "="*50)
    print("Example: Interval Width Computation")
    print("="*50)
    
    # Create dummy lower and upper bounds
    num_samples = 100
    num_dims = 33
    
    lower_bounds = torch.randn(num_samples, num_dims) * 0.5 - 1.0
    upper_bounds = lower_bounds + torch.rand(num_samples, num_dims) * 2.0
    
    print(f"\nLower bounds shape: {lower_bounds.shape}")
    print(f"Upper bounds shape: {upper_bounds.shape}")
    
    # Compute interval widths
    widths = compute_interval_width(lower_bounds, upper_bounds)
    print(f"\nInterval widths shape: {widths.shape}")
    print(f"Mean interval width: {widths.mean():.4f}")
    print(f"Std interval width: {widths.std():.4f}")
    print(f"Min interval width: {widths.min():.4f}")
    print(f"Max interval width: {widths.max():.4f}")


def example_lpips():
    """Example: Compute LPIPS distance."""
    print("\n" + "="*50)
    print("Example: LPIPS Distance Computation")
    print("="*50)
    
    try:
        # Create dummy images (normalized to [-1, 1])
        img1 = torch.randn(1, 3, 256, 256) * 0.5
        img2 = torch.randn(1, 3, 256, 256) * 0.5
        
        # Initialize LPIPS metric
        print("\nInitializing LPIPS metric...")
        lpips = LPIPSMetric(device='cuda' if torch.cuda.is_available() else 'cpu')
        
        # Compute distance
        print("Computing LPIPS distance...")
        distance = lpips.compute(img1, img2)
        print(f"LPIPS distance: {distance:.4f}")
        
    except ImportError as e:
        print(f"\nLPIPS not available: {e}")
        print("This is expected if pixel2style2pixel is not fully set up.")


if __name__ == '__main__':
    print("Extension Experiments - Example Usage")
    print("="*50)
    
    # Run examples
    example_corruptions()
    example_latent_manipulations()
    example_interval_computation()
    example_lpips()
    
    print("\n" + "="*50)
    print("Examples completed!")
    print("="*50)
    print("\nTo run full experiments, use:")
    print("  python extension_experiments/run_experiments.py --experiment all")
    print("\nOr run individual experiments:")
    print("  python extension_experiments/experiments/ablation_experiment.py")
    print("  python extension_experiments/experiments/stress_test.py")
    print("  python extension_experiments/experiments/importance_ranking.py")

