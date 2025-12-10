"""
Dataset utilities for FFHQ with corruptions.
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import glob
import os
import sys

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))

try:
    from extension_experiments.corruptions.corruption_utils import apply_corruption_level
except ImportError:
    # Fallback for direct import
    from corruptions.corruption_utils import apply_corruption_level
import torchvision.transforms as transforms


class FFHQCorruptedDataset(Dataset):
    """
    Dataset for FFHQ images with various corruption levels.
    Assumes images are generated via StyleGAN and stored with latents.
    """
    
    def __init__(self, db_path, source_transform, target_transform, 
                 corruption_level='mild', resolution=256, filelist=None):
        """
        Initialize FFHQ corrupted dataset.
        
        Args:
            db_path: Path to dataset directory
            source_transform: Transform for corrupted input images
            target_transform: Transform for target (clean) images
            corruption_level: 'mild', 'moderate', 'severe', or 'extreme'
            resolution: Image resolution
            filelist: Optional list of specific files to use
        """
        self.db_path = db_path
        self.corruption_level = corruption_level
        self.source_transform = source_transform
        self.target_transform = target_transform
        
        if filelist is None:
            # Look for images in standard structure: */img_*.png
            self.img_fnames = list(glob.glob(os.path.join(db_path, '*/*.png')))
            # Filter for image files (not latent files)
            self.img_fnames = [f for f in self.img_fnames if 'img_' in f or f.endswith('.png')]
        else:
            self.img_fnames = filelist
        
        print(f'FFHQCorruptedDataset: Found {len(self.img_fnames)} images')
        print(f'Corruption level: {corruption_level}')
    
    def __len__(self):
        return len(self.img_fnames)
    
    def img_path_to_latent_path(self, img_path):
        """Convert image path to corresponding latent path."""
        return img_path.replace('/img_', '/latents_').replace('.png', '.npz')
    
    def __getitem__(self, idx):
        """
        Get item with corruption applied.
        
        Returns:
            corrupted_image, target_image, true_style_vectors, true_wplus, img_fname
        """
        img_fname = self.img_fnames[idx]
        latent_fname = self.img_path_to_latent_path(img_fname)
        
        # Load image
        rgb_image = Image.open(img_fname).convert('RGB')
        
        # Apply target transform (clean image)
        target_image = self.target_transform(rgb_image)
        
        # Apply source transform and corruption
        if self.source_transform:
            source_image = self.source_transform(rgb_image)
        else:
            source_image = target_image.clone()
        
        # Apply corruption level
        corrupted_image = apply_corruption_level(source_image, level=self.corruption_level)
        
        # Load latents if available
        true_style_vectors = None
        true_wplus = None
        if os.path.exists(latent_fname):
            with np.load(latent_fname) as data:
                true_style_vectors = torch.Tensor(data['style_vectors']) if 'style_vectors' in data else None
                true_wplus = torch.Tensor(data['wplus']) if 'wplus' in data else None
        
        return corrupted_image, target_image, true_style_vectors, true_wplus, img_fname

