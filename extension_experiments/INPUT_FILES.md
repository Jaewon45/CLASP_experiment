# Required Input Files

This document lists all non-Python input files needed to run the extension experiments.

## Overview

For simulation/testing purposes, you need the following input files:

1. **Model files** (`.pt` files)
2. **Image files** (`.png` files) - optional if you have outputs.npz
3. **Latent files** (`.npz` files) - optional if you have outputs.npz
4. **Calibration data** (`outputs.npz` files) - **REQUIRED**
5. **Pretrained models** (`.pt` files)

## File Structure

```
{base_dir}/{exp_name}/
├── models/
│   └── {model_name}.pt                    # REQUIRED: Your trained encoder model
├── pretrained_models/
│   └── psp_celebs_super_resolution.pt      # REQUIRED: Pretrained pSp generator
└── data/
    └── calibration_set_outputs_generated_data/
        ├── {resize_factor_1}/              # e.g., '1', '16', '32'
        │   └── outputs.npz                 # REQUIRED: Calibration predictions
        ├── {resize_factor_2}/
        │   └── outputs.npz                 # REQUIRED: Calibration predictions
        └── {resize_factor_3}/
            └── outputs.npz                 # REQUIRED: Calibration predictions
```

## Required Files

### 1. Encoder Model (REQUIRED)

**File**: `{base_dir}/{exp_name}/models/{model_name}.pt`

**Example**: `assets/super_resolution/models/superres_alpha_0.1.pt`

**What it is**: Your trained quantile encoder model that predicts style vectors from images.

**How to get it**: Train using the training scripts in `pixel2style2pixel/training/`

**Contents**: PyTorch checkpoint with:
- `state_dict`: Model weights
- `opts`: Training options
- `latent_avg`: Average latent vector

### 2. Pretrained pSp Generator (REQUIRED)

**File**: `{base_dir}/{exp_name}/pretrained_models/psp_celebs_super_resolution.pt`

**What it is**: Pretrained pSp (pixel2style2pixel) model for generating images from style vectors.

**How to get it**: 
- Download from the pixel2style2pixel repository
- Or use the download script: `pixel2style2pixel/download-weights.sh`

**Note**: The code hardcodes this path in `qgan_superres_results_runner.py` line 823. You may need to adjust if your path differs.

### 3. Calibration Data - outputs.npz (REQUIRED)

**File**: `{base_dir}/{exp_name}/data/calibration_set_outputs_generated_data/{resize_factor}/outputs.npz`

**Example**: 
- `assets/super_resolution/data/calibration_set_outputs_generated_data/1/outputs.npz`
- `assets/super_resolution/data/calibration_set_outputs_generated_data/16/outputs.npz`
- `assets/super_resolution/data/calibration_set_outputs_generated_data/32/outputs.npz`

**What it is**: Aggregated predictions from running the encoder on calibration images.

**Required contents**:
```python
{
    'all_sv': np.ndarray,                    # Ground truth style vectors [N, D]
    'all_sv_hat': np.ndarray,                # Predicted style vectors [N, D]
    'all_sv_hat_lq': np.ndarray,             # Lower quantile predictions [N, D]
    'all_sv_hat_uq': np.ndarray,             # Upper quantile predictions [N, D]
    'mean_style_space_vector': np.ndarray,   # Mean vector [D]
    'min_style_space_vector': np.ndarray,    # Min vector [D]
    'max_style_space_vector': np.ndarray,    # Max vector [D]
    'style_dims': np.ndarray                 # Style dimension sizes
}
```

**How to generate**:
```bash
python extension_experiments/utils/generate_outputs_npz.py \
    --resize_factor 1 \
    --image_dir <your_images> \
    --output_path <output_path>/outputs.npz
```

**For simulation**: You can create synthetic `outputs.npz` files with random data for testing.

## Optional Files (for full pipeline)

### 4. Individual Images (OPTIONAL if you have outputs.npz)

**Files**: `*.png` image files

**Location**: Can be anywhere, but typically:
```
{base_dir}/{exp_name}/data/ffhq_generated/
├── subfolder_1/
│   ├── img_00001.png
│   ├── img_00002.png
│   └── ...
└── subfolder_2/
    └── ...
```

**What they are**: FFHQ images (real or generated via StyleGAN).

**When needed**: 
- When generating `outputs.npz` files (input to encoder)
- For visualization in experiments
- For ablation experiments that generate new images

**For simulation**: Not strictly needed if you already have `outputs.npz` files.

### 5. Individual Latent Files (OPTIONAL if you have outputs.npz)

**Files**: `latents_*.npz` files (one per image)

**Location**: Same directory as images, or parallel structure:
```
{base_dir}/{exp_name}/data/ffhq_generated/
├── subfolder_1/
│   ├── img_00001.png
│   ├── latents_00001.npz      # Corresponding latent
│   ├── img_00002.png
│   └── latents_00002.npz
```

**What they are**: Style vectors and W+ codes for each image.

**Required contents**:
```python
{
    'style_vectors': np.ndarray,  # Style space vectors
    'wplus': np.ndarray          # W+ latent codes
}
```

**When needed**:
- When generating `outputs.npz` (if you want ground truth `all_sv`)
- For ablation experiments that need true latents

**For simulation**: Not strictly needed if you already have `outputs.npz` with `all_sv`.

## Minimal Setup for Simulation

If you're just simulating/testing, you only need:

1. ✅ **Encoder model** (`.pt` file)
2. ✅ **Pretrained pSp model** (`.pt` file)
3. ✅ **outputs.npz files** (one per resize factor)

You can create synthetic `outputs.npz` files for testing:

```python
import numpy as np

# Create synthetic outputs.npz
num_samples = 1000
num_dims = 9216  # Typical style vector dimension

outputs = {
    'all_sv': np.random.randn(num_samples, num_dims),
    'all_sv_hat': np.random.randn(num_samples, num_dims),
    'all_sv_hat_lq': np.random.randn(num_samples, num_dims) - 0.1,
    'all_sv_hat_uq': np.random.randn(num_samples, num_dims) + 0.1,
    'mean_style_space_vector': np.zeros(num_dims),
    'min_style_space_vector': np.full(num_dims, -2.0),
    'max_style_space_vector': np.full(num_dims, 2.0),
    'style_dims': np.array([0, 512, 512, 512, ...])  # StyleGAN dimensions
}

np.savez('outputs.npz', **outputs)
```

## File Size Estimates

- **Encoder model** (`.pt`): ~50-200 MB
- **Pretrained pSp model** (`.pt`): ~300-500 MB
- **outputs.npz** (per resize factor): 
  - 1000 samples: ~50-100 MB
  - 10000 samples: ~500 MB - 1 GB
  - Depends on number of dimensions

## Summary

**Minimum required for experiments**:
1. Encoder model (`.pt`)
2. Pretrained pSp model (`.pt`)
3. `outputs.npz` files (one per resize factor)

**Optional (for full pipeline)**:
4. Individual image files (`.png`)
5. Individual latent files (`latents_*.npz`)

**For simulation**: You can create synthetic `outputs.npz` files and skip images/latents if you just want to test the experiment framework.

