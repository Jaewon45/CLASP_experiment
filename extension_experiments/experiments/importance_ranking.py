"""
Semantic-Factor Importance Ranking.
Rank semantic dimensions by interval width: Importance_d = W_d
"""

import os
import sys
import numpy as np
import torch
from tqdm import tqdm
import json
import pandas as pd
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))

from utils.latent_utils import compute_interval_width
from qgan_superres_results_runner import RCPS_Results_Runner


class ImportanceRankingExperiment:
    """Rank semantic dimensions by importance (interval width)."""
    
    def __init__(self, results_runner, device='auto'):
        """
        Initialize importance ranking experiment.
        
        Args:
            results_runner: RCPS_Results_Runner instance
            device: Device to run on ('auto', 'cuda', 'mps', or 'cpu')
        """
        self.results_runner = results_runner
        self.device = device
        
        # Get dimensions
        self.disent_dims = results_runner.disent_dims
        self.select_indices = results_runner.select_indices
        self.num_dims = len(self.select_indices)
        
        # Semantic factor names (from StyleGAN analysis)
        self.factor_names = [
            'hc0', 'hc1', 'hc2', 'hl3', 'hl4', 'hl5', 'hl6', 'hl7',
            'hs8', 'hs9', 'hs10', 'hs11', 'hs12', 'hs13', 'hs14', 'hs15', 'hs16',
            'm17', 'm18', 'e19', 'e20', 'e21', 'e22', 'g23'
        ]
    
    def compute_importance_scores(self, difficulty_level='easy', run_index=-1, 
                                   num_samples=None):
        """
        Compute importance scores (interval widths) for each dimension.
        
        Args:
            difficulty_level: Difficulty level to use
            run_index: Which calibration run to use
            num_samples: Number of samples to use (None = all)
        
        Returns:
            importance_scores: Importance scores per dimension (shape: num_dims,)
            dimension_indices: Actual dimension indices
        """
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
        
        # Compute average interval width per dimension (importance score)
        importance_scores = compute_interval_width(lower_edges_selected, upper_edges_selected)
        
        # Get actual dimension indices
        dimension_indices = self.disent_dims[self.select_indices]
        
        # Convert to numpy
        if isinstance(importance_scores, torch.Tensor):
            importance_scores = importance_scores.cpu().numpy()
        if isinstance(dimension_indices, torch.Tensor):
            dimension_indices = dimension_indices.cpu().numpy()
        
        return importance_scores, dimension_indices
    
    def rank_dimensions(self, difficulty_level='easy', run_index=-1, num_samples=None,
                       save_results=True, output_dir='results/importance_ranking'):
        """
        Rank dimensions by importance.
        
        Args:
            difficulty_level: Difficulty level to use
            run_index: Which calibration run to use
            num_samples: Number of samples to use (None = all)
            save_results: Whether to save results
            output_dir: Directory to save results
        
        Returns:
            ranking_df: DataFrame with ranked dimensions
        """
        print(f"Computing importance scores for {self.num_dims} dimensions...")
        
        importance_scores, dimension_indices = self.compute_importance_scores(
            difficulty_level=difficulty_level,
            run_index=run_index,
            num_samples=num_samples
        )
        
        # Create ranking
        sorted_indices = np.argsort(importance_scores)[::-1]  # Descending order
        
        # Create DataFrame
        ranking_data = []
        for rank, idx in enumerate(sorted_indices, 1):
            dim_idx = dimension_indices[idx]
            importance = importance_scores[idx]
            
            # Get factor name if available
            factor_name = None
            if idx < len(self.factor_names):
                factor_name = self.factor_names[idx]
            
            ranking_data.append({
                'rank': rank,
                'dimension_index': int(dim_idx),
                'select_index': int(self.select_indices[idx]),
                'factor_name': factor_name,
                'importance_score': float(importance)
            })
        
        ranking_df = pd.DataFrame(ranking_data)
        
        # Save results
        if save_results:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save as CSV
            csv_file = os.path.join(output_dir, 'importance_ranking.csv')
            ranking_df.to_csv(csv_file, index=False)
            print(f"Ranking saved to {csv_file}")
            
            # Save as JSON
            json_file = os.path.join(output_dir, 'importance_ranking.json')
            ranking_dict = {
                'difficulty_level': difficulty_level,
                'run_index': run_index,
                'num_samples': num_samples if num_samples else 'all',
                'ranking': ranking_df.to_dict('records')
            }
            with open(json_file, 'w') as f:
                json.dump(ranking_dict, f, indent=2)
            print(f"Results saved to {json_file}")
        
        # Print top 10
        print("\nTop 10 Most Important Dimensions:")
        print(ranking_df.head(10).to_string(index=False))
        
        return ranking_df
    
    def compare_across_difficulty_levels(self, run_index=-1, num_samples=None,
                                        save_results=True, output_dir='results/importance_ranking'):
        """
        Compare importance rankings across different difficulty levels.
        
        Args:
            run_index: Which calibration run to use
            num_samples: Number of samples to use (None = all)
            save_results: Whether to save results
            output_dir: Directory to save results
        
        Returns:
            comparison_dict: Dictionary with rankings for each difficulty level
        """
        difficulty_levels = ['easy', 'medium', 'hard']
        comparison_dict = {}
        
        print("Comparing importance rankings across difficulty levels...")
        
        for difficulty_level in tqdm(difficulty_levels, desc="Processing difficulty levels"):
            try:
                ranking_df = self.rank_dimensions(
                    difficulty_level=difficulty_level,
                    run_index=run_index,
                    num_samples=num_samples,
                    save_results=False  # Save separately below
                )
                comparison_dict[difficulty_level] = ranking_df.to_dict('records')
            except Exception as e:
                print(f"Error processing {difficulty_level}: {e}")
                comparison_dict[difficulty_level] = {'error': str(e)}
        
        # Save comparison
        if save_results:
            os.makedirs(output_dir, exist_ok=True)
            comparison_file = os.path.join(output_dir, 'importance_comparison.json')
            with open(comparison_file, 'w') as f:
                json.dump(comparison_dict, f, indent=2)
            print(f"\nComparison saved to {comparison_file}")
        
        return comparison_dict


if __name__ == '__main__':
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Run importance ranking experiment')
    parser.add_argument('--base_dir', type=str, default='assets', help='Base directory')
    parser.add_argument('--exp_name', type=str, default='super_resolution', help='Experiment name')
    parser.add_argument('--model_name', type=str, default='models/superres_alpha_0.1.pt', help='Model name')
    parser.add_argument('--difficulty_level', type=str, default='easy', help='Difficulty level')
    parser.add_argument('--run_index', type=int, default=-1, help='Calibration run index')
    parser.add_argument('--num_samples', type=int, default=None, help='Number of samples to use')
    parser.add_argument('--compare_levels', action='store_true', help='Compare across difficulty levels')
    parser.add_argument('--output_dir', type=str, default='results/importance_ranking', help='Output directory')
    
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
    
    # Run importance ranking
    ranking_exp = ImportanceRankingExperiment(results_runner)
    
    if args.compare_levels:
        comparison = ranking_exp.compare_across_difficulty_levels(
            run_index=args.run_index,
            num_samples=args.num_samples,
            output_dir=args.output_dir
        )
    else:
        ranking = ranking_exp.rank_dimensions(
            difficulty_level=args.difficulty_level,
            run_index=args.run_index,
            num_samples=args.num_samples,
            output_dir=args.output_dir
        )
    
    print("Importance ranking experiment completed!")

