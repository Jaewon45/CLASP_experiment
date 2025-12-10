"""
Corruption utilities for FFHQ experiments.
Implements various corruption types: blur, downsampling, masking, adversarial noise, block dropout.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.nn.functional import interpolate

# Import from parent project
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
from pixel2style2pixel.datasets.augmentations import BicubicDownSample

# Try to import cv2, fallback to torch-based blur
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def apply_blur(image_tensor, kernel_size=5, sigma=1.0):
    """
    Apply Gaussian blur to image tensor.
    
    Args:
        image_tensor: torch.Tensor of shape (C, H, W) or (B, C, H, W)
        kernel_size: Size of Gaussian kernel (must be odd)
        sigma: Standard deviation of Gaussian kernel
    
    Returns:
        Blurred image tensor
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Create Gaussian kernel
    if HAS_CV2:
        kernel = cv2.getGaussianKernel(kernel_size, sigma)
        kernel = kernel @ kernel.T
        kernel = torch.from_numpy(kernel).float()
    else:
        # Fallback: create Gaussian kernel using torch
        coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        kernel = g.unsqueeze(0) * g.unsqueeze(1)
    
    # Normalize
    kernel = kernel / kernel.sum()
    
    # Expand for RGB
    was_3d = image_tensor.dim() == 3
    if was_3d:
        kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
        image_tensor = image_tensor.unsqueeze(0)
    else:
        kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(image_tensor.shape[1], 1, 1, 1)
    
    kernel = kernel.to(image_tensor.device)
    
    # Apply convolution
    padding = kernel_size // 2
    blurred = F.conv2d(image_tensor, kernel, padding=padding, groups=image_tensor.shape[1])
    
    if was_3d:
        blurred = blurred.squeeze(0)
    
    return blurred


def apply_downsampling(image_tensor, factor, method='bicubic'):
    """
    Apply downsampling to image tensor.
    
    Args:
        image_tensor: torch.Tensor of shape (C, H, W) or (B, C, H, W)
        factor: Downsampling factor (e.g., 2, 8, 32, 64)
        method: 'bicubic' or 'bilinear'
    
    Returns:
        Downsampled and upsampled image tensor (same size as input)
    """
    if factor == 1:
        return image_tensor
    
    was_3d = image_tensor.dim() == 3
    if was_3d:
        image_tensor = image_tensor.unsqueeze(0)
    
    # Use BicubicDownSample from augmentations
    downsample = BicubicDownSample(factor=factor, cuda=image_tensor.device.type == 'cuda')
    
    # Downsample
    downsampled = downsample(image_tensor)
    
    # Upsample back to original size
    _, _, h, w = image_tensor.shape
    if method == 'bicubic':
        upsampled = F.interpolate(downsampled, size=(h, w), mode='bicubic', align_corners=False)
    else:
        upsampled = F.interpolate(downsampled, size=(h, w), mode='bilinear', align_corners=False)
    
    if was_3d:
        upsampled = upsampled.squeeze(0)
    
    return upsampled


def apply_masking(image_tensor, mask_ratio=0.5, mask_cent=0.5):
    """
    Apply random masking to image tensor.
    
    Args:
        image_tensor: torch.Tensor of shape (C, H, W) or (B, C, H, W)
        mask_ratio: Ratio of image to mask (0.0 to 1.0)
        mask_cent: Center value for mask (typically 0.5)
    
    Returns:
        Masked image tensor and mask tensor
    """
    was_3d = image_tensor.dim() == 3
    if was_3d:
        image_tensor = image_tensor.unsqueeze(0)
    
    batch_size = image_tensor.shape[0]
    image_size = image_tensor.shape[-1]
    
    # Create random mask
    m = torch.rand(batch_size, 1, 6, 6).to(image_tensor.device)
    m = interpolate(m, size=image_size, mode='bilinear', align_corners=False)
    
    # Threshold based on mask_ratio
    threshold = 1.0 - mask_ratio
    mask = (m < threshold).float()
    
    masked_image = image_tensor * mask
    mask_centered = mask - mask_cent
    
    if was_3d:
        masked_image = masked_image.squeeze(0)
        mask_centered = mask_centered.squeeze(0)
    
    return masked_image, mask_centered


def apply_adversarial_noise(image_tensor, epsilon=0.03, norm='inf'):
    """
    Apply adversarial noise (FGSM-style) to image tensor.
    
    Args:
        image_tensor: torch.Tensor of shape (C, H, W) or (B, C, H, W)
        epsilon: Perturbation magnitude
        norm: 'inf' for L-infinity norm, '2' for L2 norm
    
    Returns:
        Adversarially perturbed image tensor
    """
    was_3d = image_tensor.dim() == 3
    if was_3d:
        image_tensor = image_tensor.unsqueeze(0)
    
    # Generate random noise
    noise = torch.randn_like(image_tensor)
    
    if norm == 'inf':
        noise = torch.sign(noise) * epsilon
    else:  # L2 norm
        noise = noise / (noise.norm(dim=(1, 2, 3), keepdim=True) + 1e-8) * epsilon
    
    # Clip to valid range [-1, 1]
    perturbed = torch.clamp(image_tensor + noise, -1.0, 1.0)
    
    if was_3d:
        perturbed = perturbed.squeeze(0)
    
    return perturbed


def apply_block_dropout(image_tensor, block_size=32, dropout_prob=0.3):
    """
    Apply block dropout (randomly drop blocks of pixels).
    
    Args:
        image_tensor: torch.Tensor of shape (C, H, W) or (B, C, H, W)
        block_size: Size of blocks to drop
        dropout_prob: Probability of dropping each block
    
    Returns:
        Image tensor with dropped blocks
    """
    was_3d = image_tensor.dim() == 3
    if was_3d:
        image_tensor = image_tensor.unsqueeze(0)
    
    _, _, h, w = image_tensor.shape
    result = image_tensor.clone()
    
    # Create block dropout mask
    num_blocks_h = h // block_size
    num_blocks_w = w // block_size
    
    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            if np.random.rand() < dropout_prob:
                h_start = i * block_size
                h_end = min((i + 1) * block_size, h)
                w_start = j * block_size
                w_end = min((j + 1) * block_size, w)
                
                # Set block to zero (or mean value)
                result[:, :, h_start:h_end, w_start:w_end] = 0.0
    
    if was_3d:
        result = result.squeeze(0)
    
    return result


def apply_corruption_level(image_tensor, level='mild'):
    """
    Apply corruption based on predefined severity levels.
    
    Args:
        image_tensor: torch.Tensor of shape (C, H, W) or (B, C, H, W)
        level: 'mild', 'moderate', 'severe', or 'extreme'
    
    Returns:
        Corrupted image tensor
    """
    was_3d = image_tensor.dim() == 3
    if was_3d:
        image_tensor = image_tensor.unsqueeze(0)
    
    corrupted = image_tensor.clone()
    
    if level == 'mild':
        # Blur + 2x downsampling
        corrupted = apply_blur(corrupted, kernel_size=5, sigma=1.0)
        corrupted = apply_downsampling(corrupted, factor=2, method='bicubic')
    
    elif level == 'moderate':
        # 8x SR input + 50% masking
        corrupted = apply_downsampling(corrupted, factor=8, method='bicubic')
        corrupted, _ = apply_masking(corrupted, mask_ratio=0.5)
    
    elif level == 'severe':
        # 32x downsampling + 80% masking
        corrupted = apply_downsampling(corrupted, factor=32, method='bicubic')
        corrupted, _ = apply_masking(corrupted, mask_ratio=0.8)
    
    elif level == 'extreme':
        # 64x downsampling + 90% masking + adversarial noise + block dropout
        corrupted = apply_downsampling(corrupted, factor=64, method='bicubic')
        corrupted, _ = apply_masking(corrupted, mask_ratio=0.9)
        corrupted = apply_adversarial_noise(corrupted, epsilon=0.03, norm='inf')
        corrupted = apply_block_dropout(corrupted, block_size=32, dropout_prob=0.3)
    
    else:
        raise ValueError(f"Unknown corruption level: {level}")
    
    if was_3d:
        corrupted = corrupted.squeeze(0)
    
    return corrupted

