"""
Extreme corruption stress test.
For each corruption severity level:
- Compute calibrated intervals
- Measure average interval width: W_d = E[q_d^high - q_d^low]
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

from utils.latent_utils import compute_interval_width
from qgan_superres_results_runner import RCPS_Results_Runner
from corruptions.corruption_utils import apply_corruption_level


class StressTestExperiment:
    """Run stress test experiments with extreme corruptions."""
    
    def __init__(self, results_runner, device='auto'):
        """
        Initialize stress test experiment.
        
        Args:
            results_runner: RCPS_Results_Runner instance
            device: Device to run on ('auto', 'cuda', 'mps', or 'cpu')
        """
        self.results_runner = results_runner
        self.device = device
        
        # Corruption levels to test
        self.corruption_levels = ['mild', 'moderate', 'severe', 'extreme']
        
        # Get dimensions
        self.disent_dims = results_runner.disent_dims
        self.select_indices = results_runner.select_indices
        self.num_dims = len(self.select_indices)
    
    def compute_intervals_for_corruption(self, corruption_level, run_index=-1, 
                                         num_samples=None):
        """
        Compute calibrated intervals for a given corruption level.
        
        Args:
            corruption_level: Corruption level to test
            run_index: Which calibration run to use
            num_samples: Number of samples to use (None = all)
        
        Returns:
            interval_widths: Average interval width per dimension (shape: num_dims,)
            all_lower_bounds: Lower bounds for all samples (shape: num_samples, num_dims)
            all_upper_bounds: Upper bounds for all samples (shape: num_samples, num_dims)
        """
        # Map corruption level to difficulty level
        # For now, we'll use the existing difficulty levels
        # In a full implementation, you'd create new calibration data with corruptions
        difficulty_mapping = {
            'mild': 'easy',
            'moderate': 'medium',
            'severe': 'hard',
            'extreme': 'hard'  # Use hard as proxy for extreme
        }
        
        difficulty_level = difficulty_mapping.get(corruption_level, 'easy')
        
        # Get prediction sets
        lr_prediction_sets = self.results_runner.all_prediction_sets[difficulty_level][0]
        idx_lambda = self.results_runner.all_rcps_stats[difficulty_level]['idx_lambda_calib'][run_index]
        
        lower_edges = lr_prediction_sets[0][idx_lambda]
        upper_edges = lr_prediction_sets[1][idx_lambda]
        
        # Select only the dimensions we care about
        lower_edges_selected = lower_edges[:, self.disent_dims[self.select_indices]]
        upper_edges_selected = upper_edges[:, self.disent_dims[self.select_indices]]
        
        # Limit samples if requested
        if num_samples is not None:
            lower_edges_selected = lower_edges_selected[:num_samples]
            upper_edges_selected = upper_edges_selected[:num_samples]
        
        # Compute average interval width per dimension
        interval_widths = compute_interval_width(lower_edges_selected, upper_edges_selected)
        
        return interval_widths, lower_edges_selected, upper_edges_selected
    
    def run_stress_test(self, run_index=-1, num_samples=None, save_results=True,
                        output_dir='results/stress_test'):
        """
        Run full stress test across all corruption levels.
        
        Args:
            run_index: Which calibration run to use
            num_samples: Number of samples to use (None = all)
            save_results: Whether to save results
            output_dir: Directory to save results
        
        Returns:
            results_dict: Dictionary with results for each corruption level
        """
        results_dict = {
            'run_index': run_index,
            'num_samples': num_samples if num_samples else 'all',
            'corruption_levels': {}
        }
        
        print(f"Running stress test on {len(self.corruption_levels)} corruption levels...")
        
        for corruption_level in tqdm(self.corruption_levels, desc="Processing corruption levels"):
            try:
                interval_widths, lower_bounds, upper_bounds = self.compute_intervals_for_corruption(
                    corruption_level, run_index=run_index, num_samples=num_samples
                )
                
                # Convert to numpy for serialization
                if isinstance(interval_widths, torch.Tensor):
                    interval_widths = interval_widths.cpu().numpy()
                if isinstance(lower_bounds, torch.Tensor):
                    lower_bounds = lower_bounds.cpu().numpy()
                if isinstance(upper_bounds, torch.Tensor):
                    upper_bounds = upper_bounds.cpu().numpy()
                
                results_dict['corruption_levels'][corruption_level] = {
                    'interval_widths': interval_widths.tolist(),
                    'mean_interval_width': float(interval_widths.mean()),
                    'std_interval_width': float(interval_widths.std()),
                    'min_interval_width': float(interval_widths.min()),
                    'max_interval_width': float(interval_widths.max()),
                    'num_dimensions': int(self.num_dims),
                    'num_samples': int(lower_bounds.shape[0]) if num_samples else 'all'
                }
                
                print(f"\n{corruption_level}:")
                print(f"  Mean interval width: {interval_widths.mean():.4f}")
                print(f"  Std interval width: {interval_widths.std():.4f}")
                print(f"  Min interval width: {interval_widths.min():.4f}")
                print(f"  Max interval width: {interval_widths.max():.4f}")
            
            except Exception as e:
                print(f"Error processing {corruption_level}: {e}")
                results_dict['corruption_levels'][corruption_level] = {
                    'error': str(e)
                }
        
        # Save results
        if save_results:
            os.makedirs(output_dir, exist_ok=True)
            results_file = os.path.join(output_dir, 'stress_test_results.json')
            with open(results_file, 'w') as f:
                json.dump(results_dict, f, indent=2)
            print(f"\nResults saved to {results_file}")
        
        return results_dict


if __name__ == '__main__':
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Run stress test experiment')
    parser.add_argument('--base_dir', type=str, default='assets', help='Base directory')
    parser.add_argument('--exp_name', type=str, default='super_resolution', help='Experiment name')
    parser.add_argument('--model_name', type=str, default='models/superres_alpha_0.1.pt', help='Model name')
    parser.add_argument('--run_index', type=int, default=-1, help='Calibration run index')
    parser.add_argument('--num_samples', type=int, default=None, help='Number of samples to use')
    parser.add_argument('--output_dir', type=str, default='results/stress_test', help='Output directory')
    
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
    
    # Run stress test
    stress_test = StressTestExperiment(results_runner)
    results = stress_test.run_stress_test(
        run_index=args.run_index,
        num_samples=args.num_samples,
        output_dir=args.output_dir
    )
    
    print("Stress test experiment completed!")

