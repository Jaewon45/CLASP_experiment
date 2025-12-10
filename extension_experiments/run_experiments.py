"""
Main script to run all extension experiments.
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

from qgan_superres_results_runner import RCPS_Results_Runner
from experiments.ablation_experiment import AblationExperiment
from experiments.stress_test import StressTestExperiment
from experiments.importance_ranking import ImportanceRankingExperiment


def main():
    parser = argparse.ArgumentParser(description='Run extension experiments')
    parser.add_argument('--base_dir', type=str, default='assets', help='Base directory')
    parser.add_argument('--exp_name', type=str, default='super_resolution', help='Experiment name')
    parser.add_argument('--model_name', type=str, default='models/superres_alpha_0.1.pt', help='Model name')
    parser.add_argument('--experiment', type=str, choices=['ablation', 'stress_test', 'importance', 'all'],
                       default='all', help='Which experiment to run')
    parser.add_argument('--output_dir', type=str, default='extension_experiments/results', help='Output directory')
    parser.add_argument('--run_index', type=int, default=-1, help='Calibration run index')
    parser.add_argument('--image_index', type=int, default=0, help='Image index for ablation')
    parser.add_argument('--difficulty_level', type=str, default='easy', help='Difficulty level')
    parser.add_argument('--num_samples', type=int, default=None, help='Number of samples to use')
    
    args = parser.parse_args()
    
    # Initialize results runner
    print("Initializing results runner...")
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
    print("Computing losses and calibrating...")
    if not hasattr(results_runner, 'all_rcps_stats') or len(results_runner.all_rcps_stats['easy']) == 0:
        results_runner.compute_losses_prediction_sets()
        results_runner.calibrate_all_difficulty_levels(total_runs=100)
    
    # Run experiments
    if args.experiment in ['ablation', 'all']:
        print("\n" + "="*50)
        print("Running Ablation Experiment")
        print("="*50)
        ablation = AblationExperiment(results_runner)
        ablation_results = ablation.run_ablation(
            image_index=args.image_index,
            run_index=args.run_index,
            difficulty_level=args.difficulty_level,
            output_dir=os.path.join(args.output_dir, 'ablation')
        )
        print("Ablation experiment completed!")
    
    if args.experiment in ['stress_test', 'all']:
        print("\n" + "="*50)
        print("Running Stress Test Experiment")
        print("="*50)
        stress_test = StressTestExperiment(results_runner)
        stress_results = stress_test.run_stress_test(
            run_index=args.run_index,
            num_samples=args.num_samples,
            output_dir=os.path.join(args.output_dir, 'stress_test')
        )
        print("Stress test experiment completed!")
    
    if args.experiment in ['importance', 'all']:
        print("\n" + "="*50)
        print("Running Importance Ranking Experiment")
        print("="*50)
        ranking_exp = ImportanceRankingExperiment(results_runner)
        ranking = ranking_exp.rank_dimensions(
            difficulty_level=args.difficulty_level,
            run_index=args.run_index,
            num_samples=args.num_samples,
            output_dir=os.path.join(args.output_dir, 'importance_ranking')
        )
        print("Importance ranking experiment completed!")
    
    print("\n" + "="*50)
    print("All experiments completed!")
    print("="*50)


if __name__ == '__main__':
    main()

