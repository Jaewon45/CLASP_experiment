# Extension Experiments: FFHQ Corruption Analysis

This extension implements comprehensive experiments for analyzing uncertainty quantification in GAN-based super-resolution under various corruption conditions.

## Overview

The extension includes:
- **Corruption utilities**: Blur, downsampling, masking, adversarial noise, block dropout
- **Ablation experiments**: Latent dimension perturbation analysis
- **Stress tests**: Extreme corruption analysis
- **Importance ranking**: Semantic factor importance analysis

## Quick Reference

- **[QUICK_START.md](QUICK_START.md)**: ⚡ **START HERE** - Quick summary of required files (5 files total!)
- **[M1_MAC_SETUP.md](M1_MAC_SETUP.md)**: 🍎 **For M1/M2/M3 Mac users** - Device setup and MPS support
- **[ERROR_HANDLING.md](ERROR_HANDLING.md)**: Details on error handling and fallbacks
- **[DATA_STRUCTURE.md](DATA_STRUCTURE.md)**: Data storage requirements and directory structure
- **[MEMORY_MANAGEMENT.md](MEMORY_MANAGEMENT.md)**: How to adjust `num_samples` for memory management
- **[INPUT_FILES.md](INPUT_FILES.md)**: Complete list of required input files (for simulation/testing)

## Structure

```
extension_experiments/
├── corruptions/
│   ├── __init__.py
│   └── corruption_utils.py      # Corruption implementations
├── experiments/
│   ├── ablation_experiment.py   # Latent dimension ablation
│   ├── stress_test.py            # Extreme corruption stress test
│   └── importance_ranking.py     # Semantic factor ranking
├── utils/
│   ├── __init__.py
│   ├── lpips_utils.py            # LPIPS metric wrapper
│   ├── latent_utils.py            # Latent manipulation utilities
│   └── dataset_utils.py          # FFHQ dataset with corruptions
├── results/                       # Output directory
├── run_experiments.py             # Main runner script
└── README.md                      # This file
```

## Corruption Levels

### A. Corruption Severity Levels

1. **Mild**: Blur + 2x downsampling
2. **Moderate**: 8x SR input + 50% masking
3. **Severe**: 32x downsampling + 80% masking
4. **Extreme**: 64x downsampling + 90% masking + adversarial noise + block dropout

## Experiments

### B. Ablation Experiments

For each latent dimension `d`:
- Apply perturbations:
  - Gaussian noise to dimension `d`
  - Zero-out dimension `d`
  - Random re-sample across the prior
- Recompute semantic intervals and endpoint visualizations
- Measure stability via: `Δ_d = LPIPS(G(Z_d^low), G(Z_d^high))`

**Usage:**
```bash
python extension_experiments/experiments/ablation_experiment.py \
    --base_dir assets \
    --exp_name super_resolution \
    --model_name models/superres_alpha_0.1.pt \
    --image_index 0 \
    --run_index -1 \
    --difficulty_level easy \
    --output_dir results/ablation
```

### C. Extreme Corruption Stress Test

For each corruption severity level:
- Compute calibrated intervals
- Measure average interval width: `W_d = E[q_d^high - q_d^low]`

**Usage:**
```bash
python extension_experiments/experiments/stress_test.py \
    --base_dir assets \
    --exp_name super_resolution \
    --model_name models/superres_alpha_0.1.pt \
    --run_index -1 \
    --num_samples 1000 \
    --output_dir results/stress_test
```

### D. Semantic-Factor Importance Ranking

Rank semantic dimensions by interval width:
- `Importance_d = W_d`

**Usage:**
```bash
python extension_experiments/experiments/importance_ranking.py \
    --base_dir assets \
    --exp_name super_resolution \
    --model_name models/superres_alpha_0.1.pt \
    --difficulty_level easy \
    --run_index -1 \
    --output_dir results/importance_ranking
```

Compare across difficulty levels:
```bash
python extension_experiments/experiments/importance_ranking.py \
    --base_dir assets \
    --exp_name super_resolution \
    --model_name models/superres_alpha_0.1.pt \
    --compare_levels \
    --output_dir results/importance_ranking
```

## Running All Experiments

Use the main runner script:

```bash
python extension_experiments/run_experiments.py \
    --base_dir assets \
    --exp_name super_resolution \
    --model_name models/superres_alpha_0.1.pt \
    --experiment all \
    --output_dir extension_experiments/results
```

## Dependencies

The extension relies on:
- Original CLASP codebase (`qgan_superres_results_runner.py`, `all_utils.py`, etc.)
- `pixel2style2pixel` package (for LPIPS, models, etc.)
- Standard scientific Python stack (numpy, torch, PIL, etc.)

## Output Format

### Ablation Results
- `ablation_results_{image_index}.json`: Full results with stability scores
- `visualizations/dim_{dim_idx}/{pert_type}_{low/high}.png`: Endpoint visualizations

### Stress Test Results
- `stress_test_results.json`: Interval widths for each corruption level

### Importance Ranking Results
- `importance_ranking.csv`: Ranked dimensions with importance scores
- `importance_ranking.json`: Full results in JSON format
- `importance_comparison.json`: Comparison across difficulty levels (if `--compare_levels`)

## Notes

1. **Data Requirements**: The experiments assume FFHQ data generated via StyleGAN, stored with corresponding latent vectors in `.npz` format. See [DATA_STRUCTURE.md](DATA_STRUCTURE.md) for detailed storage requirements.

   **Important**: The code does NOT automatically generate `.npz` files from images. You must generate them separately:
   - Individual `latents_*.npz` files: Created when generating images with StyleGAN
   - `outputs.npz` files: Created by running the encoder on images (use `utils/generate_outputs_npz.py`)

2. **Calibration**: Experiments require calibrated prediction sets. The runner will automatically compute and calibrate if needed. Calibration data should be stored in `{base_dir}/{exp_name}/data/calibration_set_outputs_generated_data/{resize_factor}/outputs.npz`.

3. **Device**: Defaults to automatic detection. On M1/M2 Macs, uses MPS (Metal Performance Shaders). See [M1_MAC_SETUP.md](M1_MAC_SETUP.md) for details.

4. **Memory**: Large-scale experiments may require significant GPU memory. Adjust `num_samples` if needed. See [MEMORY_MANAGEMENT.md](MEMORY_MANAGEMENT.md) for details on how to use this parameter.

5. **Error Handling**: The framework includes error handling and fallbacks. See [ERROR_HANDLING.md](ERROR_HANDLING.md) for details.

## Future Extensions

Potential additions:
- Support for additional corruption types
- Batch processing for multiple images
- Visualization utilities for comparing results
- Statistical significance testing
- Integration with wandb/tensorboard for logging

