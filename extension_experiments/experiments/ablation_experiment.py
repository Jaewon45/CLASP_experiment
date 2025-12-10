"""
Ablation experiment: Apply perturbations to each latent dimension and measure stability.
For each latent dimension d:
- Apply perturbations: Gaussian noise, zero-out, random re-sample
- Recompute semantic intervals and endpoint visualizations
- Measure stability via: Δ_d = LPIPS(G(Z_d^low), G(Z_d^high))
"""

import os
import sys
import numpy as np
import torch
from tqdm import tqdm
import json
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))

from utils.lpips_utils import LPIPSMetric
from utils.latent_utils import (
    apply_gaussian_noise_to_dim,
    zero_out_dim,
    random_resample_dim
)
from qgan_superres_results_runner import RCPS_Results_Runner


def get_device(device_str='auto'):
    """Get appropriate device, handling M1/M2 Macs."""
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
            return torch.device('cpu')
    elif device_str == 'cuda':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        return torch.device('cpu')


class AblationExperiment:
    """Run ablation experiments on latent dimensions."""
    
    def __init__(self, results_runner, device='auto'):
        """
        Initialize ablation experiment.
        
        Args:
            results_runner: RCPS_Results_Runner instance
            device: Device to run on ('auto', 'cuda', 'mps', or 'cpu')
        """
        self.results_runner = results_runner
        self.device_obj = get_device(device)
        self.device = str(self.device_obj)
        self.lpips_metric = LPIPSMetric(device=device)
        
        # Get style dimensions from results runner
        self.style_dims = results_runner.style_dims
        self.disent_dims = results_runner.disent_dims
        self.select_indices = results_runner.select_indices
        
        # Get number of dimensions to test
        self.num_dims = len(self.select_indices)
    
    def get_image_from_latent(self, latent_code):
        """Generate image from latent code using pretrained model."""
        # Convert to numpy if needed
        if isinstance(latent_code, torch.Tensor):
            latent_code = latent_code.detach().cpu().numpy()
        
        spv_list = self.results_runner.stylespace_flat_to_list(
            latent_code[None, :] if latent_code.ndim == 1 else latent_code,
            style_dims=self.results_runner.style_dims
        )
        
        with torch.no_grad():
            im, _ = self.results_runner.pretrained_model.decoder.forward_with_style_vectors(
                latent_codes=torch.randn(1, 512).to(self.device_obj),
                style_vectors=spv_list,
                input_is_latent=True,
                randomize_noise=False,
                return_latents=True,
                modulate=False
            )
        return im
    
    def tensor2im(self, t):
        """Convert tensor to image array."""
        t = t.detach().cpu().numpy().squeeze().transpose((1, 2, 0))
        t = (t + 1) * 127.5
        t = np.clip(t, 0, 255).astype(np.uint8)
        return t
    
    def compute_stability_for_dim(self, base_latent, dim_idx, perturbation_type='gaussian', 
                                   noise_std=0.1, run_index=-1, difficulty_level='easy'):
        """
        Compute stability metric for a specific dimension.
        
        Args:
            base_latent: Base latent code
            dim_idx: Index of dimension to perturb (in select_indices)
            perturbation_type: 'gaussian', 'zero', or 'resample'
            noise_std: Standard deviation for Gaussian noise
            run_index: Which calibration run to use
            difficulty_level: Difficulty level for getting intervals
        
        Returns:
            stability_score: LPIPS distance between low and high endpoints
            low_image: Image at lower bound
            high_image: Image at upper bound
        """
        # Get calibrated intervals for this dimension
        actual_dim = self.disent_dims[self.select_indices[dim_idx]]
        
        # Get prediction sets
        lr_prediction_sets = self.results_runner.all_prediction_sets[difficulty_level][0]
        idx_lambda = self.results_runner.all_rcps_stats[difficulty_level]['idx_lambda_calib'][run_index]
        
        # For ablation, we'll create perturbed versions
        if perturbation_type == 'gaussian':
            low_latent = apply_gaussian_noise_to_dim(base_latent, actual_dim, noise_std=-noise_std)
            high_latent = apply_gaussian_noise_to_dim(base_latent, actual_dim, noise_std=noise_std)
        elif perturbation_type == 'zero':
            low_latent = zero_out_dim(base_latent, actual_dim)
            high_latent = base_latent  # Keep original as high
        elif perturbation_type == 'resample':
            low_latent = random_resample_dim(base_latent, actual_dim, prior_mean=-1.0, prior_std=0.5)
            high_latent = random_resample_dim(base_latent, actual_dim, prior_mean=1.0, prior_std=0.5)
        else:
            raise ValueError(f"Unknown perturbation type: {perturbation_type}")
        
        # Generate images
        low_image = self.get_image_from_latent(low_latent)
        high_image = self.get_image_from_latent(high_latent)
        
        # Compute LPIPS distance
        stability_score = self.lpips_metric.compute(low_image, high_image)
        
        return stability_score, low_image, high_image
    
    def run_ablation(self, image_index=0, run_index=-1, difficulty_level='easy',
                     perturbation_types=['gaussian', 'zero', 'resample'], 
                     noise_std=0.1, save_results=True, output_dir='results/ablation'):
        """
        Run full ablation experiment.
        
        Args:
            image_index: Index of image to use
            run_index: Which calibration run to use
            difficulty_level: Difficulty level
            perturbation_types: List of perturbation types to test
            noise_std: Standard deviation for Gaussian noise
            save_results: Whether to save results
            output_dir: Directory to save results
        
        Returns:
            results_dict: Dictionary with results for each dimension and perturbation type
        """
        # Get base latent code
        lr_stats = self.results_runner.results_dict[difficulty_level]
        base_latent = lr_stats['allz_hat_np'][image_index]
        
        results_dict = {
            'image_index': image_index,
            'run_index': run_index,
            'difficulty_level': difficulty_level,
            'perturbation_types': perturbation_types,
            'noise_std': noise_std,
            'dimensions': {}
        }
        
        print(f"Running ablation experiment on {self.num_dims} dimensions...")
        
        for dim_idx in tqdm(range(self.num_dims), desc="Processing dimensions"):
            dim_results = {}
            actual_dim = self.disent_dims[self.select_indices[dim_idx]]
            
            for pert_type in perturbation_types:
                try:
                    stability, low_im, high_im = self.compute_stability_for_dim(
                        base_latent, dim_idx, perturbation_type=pert_type,
                        noise_std=noise_std, run_index=run_index,
                        difficulty_level=difficulty_level
                    )
                    
                    dim_results[pert_type] = {
                        'stability_score': float(stability),
                        'dimension_index': int(actual_dim)
                    }
                    
                    # Save visualization if requested
                    if save_results:
                        vis_dir = os.path.join(output_dir, 'visualizations', f'dim_{actual_dim}')
                        os.makedirs(vis_dir, exist_ok=True)
                        
                        # Save images
                        from PIL import Image
                        Image.fromarray(self.tensor2im(low_im)).save(
                            os.path.join(vis_dir, f'{pert_type}_low.png')
                        )
                        Image.fromarray(self.tensor2im(high_im)).save(
                            os.path.join(vis_dir, f'{pert_type}_high.png')
                        )
                
                except Exception as e:
                    print(f"Error processing dim {actual_dim} with {pert_type}: {e}")
                    dim_results[pert_type] = {'stability_score': None, 'error': str(e)}
            
            results_dict['dimensions'][f'dim_{actual_dim}'] = dim_results
        
        # Save results
        if save_results:
            os.makedirs(output_dir, exist_ok=True)
            results_file = os.path.join(output_dir, f'ablation_results_{image_index}.json')
            with open(results_file, 'w') as f:
                json.dump(results_dict, f, indent=2)
            print(f"Results saved to {results_file}")
        
        return results_dict


if __name__ == '__main__':
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Run ablation experiment')
    parser.add_argument('--base_dir', type=str, default='assets', help='Base directory')
    parser.add_argument('--exp_name', type=str, default='super_resolution', help='Experiment name')
    parser.add_argument('--model_name', type=str, default='models/superres_alpha_0.1.pt', help='Model name')
    parser.add_argument('--image_index', type=int, default=0, help='Image index to test')
    parser.add_argument('--run_index', type=int, default=-1, help='Calibration run index')
    parser.add_argument('--difficulty_level', type=str, default='easy', help='Difficulty level')
    parser.add_argument('--output_dir', type=str, default='results/ablation', help='Output directory')
    
    args = parser.parse_args()
    
    # Initialize results runner
    resize_factors = [1, 16, 32]
    difficulty_levels = ['easy', 'medium', 'hard']
    
    results_runner = RCPS_Results_Runner(
        base_dir=args.base_dir,
        exp_name=args.exp_name,
        model_name=args.model_name,
        resize_factors=resize_factors,
        difficulty_levels=difficulty_levels,
        norm_scheme='mean'
    )
    
    # Compute losses and calibrate if needed
    if not hasattr(results_runner, 'all_rcps_stats') or len(results_runner.all_rcps_stats['easy']) == 0:
        results_runner.compute_losses_prediction_sets()
        results_runner.calibrate_all_difficulty_levels(total_runs=100)
    
    # Run ablation
    ablation = AblationExperiment(results_runner)
    results = ablation.run_ablation(
        image_index=args.image_index,
        run_index=args.run_index,
        difficulty_level=args.difficulty_level,
        output_dir=args.output_dir
    )
    
    print("Ablation experiment completed!")

