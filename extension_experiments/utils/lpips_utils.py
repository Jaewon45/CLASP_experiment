"""
LPIPS metric utilities for measuring image similarity.
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

try:
    from pixel2style2pixel.criteria.lpips.lpips import LPIPS
    HAS_LPIPS = True
except (ImportError, ModuleNotFoundError):
    # Fallback if import fails
    LPIPS = None
    HAS_LPIPS = False


def get_device(device_str='auto'):
    """
    Get the appropriate device, handling M1/M2 Macs.
    
    Args:
        device_str: 'auto', 'cuda', 'mps', or 'cpu'
    
    Returns:
        device: torch.device object
    """
    if device_str == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    elif device_str == 'mps':
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            print("MPS not available, falling back to CPU")
            return torch.device('cpu')
    elif device_str == 'cuda':
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            print("CUDA not available, falling back to CPU")
            return torch.device('cpu')
    else:
        return torch.device('cpu')


class LPIPSMetric:
    """Wrapper for LPIPS metric computation."""
    
    def __init__(self, device='auto', net_type='alex'):
        """
        Initialize LPIPS metric.
        
        Args:
            device: Device to run on ('auto', 'cuda', 'mps', or 'cpu')
            net_type: Network type ('alex', 'vgg', 'squeeze')
        """
        self.device_obj = get_device(device)
        self.device = str(self.device_obj)
        
        if not HAS_LPIPS or LPIPS is None:
            raise ImportError("LPIPS module not found. Please check pixel2style2pixel installation.")
        
        # LPIPS class may hardcode 'cuda', so we need to handle device conversion
        self.lpips = LPIPS(net_type=net_type, version='0.1')
        self.lpips.eval()
        
        # Move to appropriate device (handles MPS)
        try:
            self.lpips.to(self.device_obj)
        except Exception as e:
            # If device conversion fails (e.g., MPS not supported), fall back to CPU
            print(f"Warning: Could not move LPIPS to {self.device_obj}, using CPU: {e}")
            self.device_obj = torch.device('cpu')
            self.device = 'cpu'
            self.lpips.to(self.device_obj)
    
    def compute(self, img1, img2):
        """
        Compute LPIPS distance between two images.
        
        Args:
            img1: torch.Tensor of shape (B, C, H, W) or (C, H, W), normalized to [-1, 1]
            img2: torch.Tensor of shape (B, C, H, W) or (C, H, W), normalized to [-1, 1]
        
        Returns:
            LPIPS distance (scalar or tensor)
        """
        # Ensure images are in correct format
        if img1.dim() == 3:
            img1 = img1.unsqueeze(0)
        if img2.dim() == 3:
            img2 = img2.unsqueeze(0)
        
        # Ensure images are on correct device
        img1 = img1.to(self.device_obj)
        img2 = img2.to(self.device_obj)
        
        # Ensure images are in [-1, 1] range
        img1 = torch.clamp(img1, -1.0, 1.0)
        img2 = torch.clamp(img2, -1.0, 1.0)
        
        with torch.no_grad():
            distance = self.lpips(img1, img2, reduction='mean')
        
        return distance.item() if isinstance(distance, torch.Tensor) else distance
    
    def compute_batch(self, img1_batch, img2_batch):
        """
        Compute LPIPS distance for batches of images.
        
        Args:
            img1_batch: torch.Tensor of shape (B, C, H, W)
            img2_batch: torch.Tensor of shape (B, C, H, W)
        
        Returns:
            Average LPIPS distance
        """
        return self.compute(img1_batch, img2_batch)

