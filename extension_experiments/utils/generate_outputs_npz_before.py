"""
Script to generate outputs.npz files from images.
This script runs the encoder model on images and saves predictions to outputs.npz format.
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from argparse import Namespace

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))

from pixel2style2pixel.configs import transforms_config
from pixel2style2pixel.models.psp import pSp
from pixel2style2pixel.models.encoders import psp_encoders
from dataset_utils import RGBSuperResGeneratedDataset


def get_device(device_str='auto'):
    if device_str == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')

    if device_str.startswith('cuda'):
        if torch.cuda.is_available():
            if ':' in device_str:
                gpu_id = int(device_str.split(':')[1])
                if gpu_id < torch.cuda.device_count():
                    return torch.device(f'cuda:{gpu_id}')
            return torch.device('cuda')
        return torch.device('cpu')

    if device_str == 'mps':
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')

    return torch.device('cpu')



def generate_outputs_npz(
    base_dir,
    exp_name,
    model_name,
    resize_factor,
    image_dir,
    output_path,
    device='auto',
    batch_size=32
):
    """
    Generate outputs.npz file from images using the encoder model.
    
    Args:
        base_dir: Base directory for experiment
        exp_name: Experiment name
        model_name: Name of encoder model file
        resize_factor: Downsampling factor (1, 16, 32, etc.)
        image_dir: Directory containing images (and latents_*.npz files)
        output_path: Path to save outputs.npz
        device: Device to run on ('auto', 'cuda', 'mps', or 'cpu')
        batch_size: Batch size for inference
    """
    # Get appropriate device
    device_obj = get_device(device)
    print(f"Using device: {device_obj}")
    
    # Construct model path
    model_path = os.path.join(base_dir, exp_name, model_name)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"Loading model from {model_path}")

    ckpt = torch.load(model_path, map_location='cpu')
    encoder_opts = ckpt['opts']
    encoder_opts['checkpoint_path'] = model_path
    encoder_opts = Namespace(**encoder_opts)

    # Build full pSp network
    full_model = pSp(encoder_opts)
    full_model.load_state_dict(ckpt['state_dict'], strict=True)
    full_model.eval().to(device_obj)

    # Extract the encoder (this matches the checkpoint architecture)
    if hasattr(full_model, 'encoder'):
        encoder_net = full_model.encoder
    elif hasattr(full_model, 'encoder_q'):   # Quantile version
        encoder_net = full_model.encoder_q
    else:
        raise RuntimeError("Encoder module not found in pSp model")

    encoder_net.eval().to(device_obj)

    print("Model loaded successfully (encoder extracted from pSp)")

    
    # Setup transforms
    encoder_opts.resize_factors = str(resize_factor)
    transform_obj = transforms_config.SuperResTransforms(encoder_opts).get_transforms()
    
    # Create dataset
    dataset = RGBSuperResGeneratedDataset(
        db_path=image_dir,
        source_transform=transform_obj['transform_source'],
        target_transform=transform_obj['transform_gt_train'],
        resolution=256
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Dataset loaded: {len(dataset)} images")
    
    # Collect predictions
    all_sv = []  # Ground truth style vectors
    all_sv_hat = []  # Predicted style vectors
    all_sv_hat_lq = []  # Lower quantile predictions
    all_sv_hat_uq = []  # Upper quantile predictions
    
    # Statistics for normalization
    all_style_vectors = []
    
    print("Running inference...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
            input_image, output_image, true_style_vectors, true_wplus, img_fnames = batch
            
            input_image = input_image.to(device_obj).float()
            
            # Run encoder
            predictions = encoder_net.forward(input_image)
            
            # Extract predictions (assuming shape: [B, 3, D] where 3 = [lq, mean, uq])
            predicted_spv_lq = predictions[:, 0, :].cpu().numpy()  # Lower quantile
            predicted_spv = predictions[:, 1, :].cpu().numpy()  # Mean prediction
            predicted_spv_uq = predictions[:, 2, :].cpu().numpy()  # Upper quantile
            
            # Collect ground truth if available
            if true_style_vectors is not None:
                for sv in true_style_vectors:
                    if sv is not None and not torch.isnan(sv).any():
                        sv_np = sv.cpu().numpy()
                        all_sv.append(sv_np.flatten())
                        all_style_vectors.append(sv_np.flatten())
            
            # Collect predictions
            for i in range(predicted_spv.shape[0]):
                all_sv_hat.append(predicted_spv[i])
                all_sv_hat_lq.append(predicted_spv_lq[i])
                all_sv_hat_uq.append(predicted_spv_uq[i])
    
    print(f"Processed {len(all_sv_hat)} images")
    
    # Convert to numpy arrays
    if len(all_sv) > 0:
        all_sv = np.vstack(all_sv)
    else:
        # If no ground truth, create dummy array
        all_sv = np.zeros((len(all_sv_hat), all_sv_hat[0].shape[0]))
        print("Warning: No ground truth style vectors found. Using zeros.")
    
    all_sv_hat = np.vstack(all_sv_hat)
    all_sv_hat_lq = np.vstack(all_sv_hat_lq)
    all_sv_hat_uq = np.vstack(all_sv_hat_uq)
    
    # Compute statistics
    if len(all_style_vectors) > 0:
        all_style_vectors = np.vstack(all_style_vectors)
        mean_style_space_vector = all_style_vectors.mean(axis=0)
        min_style_space_vector = all_style_vectors.min(axis=0)
        max_style_space_vector = all_style_vectors.max(axis=0)
    else:
        # Use predictions to estimate statistics
        mean_style_space_vector = all_sv_hat.mean(axis=0)
        min_style_space_vector = all_sv_hat.min(axis=0)
        max_style_space_vector = all_sv_hat.max(axis=0)
    
    # Get style dimensions from model or use default
    # This should match the encoder architecture
    style_dims = getattr(encoder_opts, 'style_dims', None)
    if style_dims is None:
        # Default StyleGAN2 style dimensions
        style_dims = np.array([
            0, 512, 512, 512, 512, 512, 512, 512, 512, 512,
            512, 512, 512, 512, 512, 512, 256, 256, 256, 128,
            128, 128, 64, 64, 64, 32, 32,
        ])
    
    # Save to outputs.npz
    print(f"Saving to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    np.savez(
        output_path,
        all_sv=all_sv,
        all_sv_hat=all_sv_hat,
        all_sv_hat_lq=all_sv_hat_lq,
        all_sv_hat_uq=all_sv_hat_uq,
        mean_style_space_vector=mean_style_space_vector,
        min_style_space_vector=min_style_space_vector,
        max_style_space_vector=max_style_space_vector,
        style_dims=style_dims
    )
    
    print(f"Successfully saved outputs.npz with {len(all_sv_hat)} samples")
    print(f"  - Ground truth vectors: {all_sv.shape}")
    print(f"  - Predicted vectors: {all_sv_hat.shape}")
    print(f"  - Lower quantile: {all_sv_hat_lq.shape}")
    print(f"  - Upper quantile: {all_sv_hat_uq.shape}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate outputs.npz from images')
    parser.add_argument('--base_dir', type=str, default='assets', help='Base directory')
    parser.add_argument('--exp_name', type=str, default='super_resolution', help='Experiment name')
    parser.add_argument('--model_name', type=str, default='models/superres_alpha_0.1.pt', help='Model name')
    parser.add_argument('--resize_factor', type=int, required=True, help='Resize factor (1, 16, 32, etc.)')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing images')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save outputs.npz')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--device', type=str, default='auto', 
                       help='Device: auto (detect), cuda, mps (Apple Silicon), or cpu')
    
    args = parser.parse_args()
    
    generate_outputs_npz(
        base_dir=args.base_dir,
        exp_name=args.exp_name,
        model_name=args.model_name,
        resize_factor=args.resize_factor,
        image_dir=args.image_dir,
        output_path=args.output_path,
        device=args.device,
        batch_size=args.batch_size
    )

