# Quick Start: Required Input Files

This is a **quick reference** for what non-Python files you need to run the extension experiments.

## TL;DR - Minimum Required Files

For **simulation/testing**, you only need **5 files**:

1. ✅ **Encoder model** (`.pt` file) - 1 file
2. ✅ **Pretrained pSp generator** (`.pt` file) - 1 file  
3. ✅ **outputs.npz files** (`.npz` files) - 3 files (one per resize factor: 1, 16, 32)

**Total: 5 files** (no images needed if you have outputs.npz!)

## Detailed File List

### Required Files (5 total)

#### 1. Encoder Model (1 file)
- **File**: `{base_dir}/{exp_name}/models/{model_name}.pt`
- **Example**: `assets/super_resolution/models/superres_alpha_0.1.pt`
- **Size**: ~50-200 MB
- **What it is**: Your trained quantile encoder model

#### 2. Pretrained pSp Generator (1 file)
- **File**: `{base_dir}/{exp_name}/pretrained_models/psp_celebs_super_resolution.pt`
- **Example**: `assets/super_resolution/pretrained_models/psp_celebs_super_resolution.pt`
- **Size**: ~300-500 MB
- **How to get**: Download from [pixel2style2pixel repository](https://github.com/eladrich/pixel2style2pixel) (Super Resolution model)

#### 3. Calibration Data - outputs.npz (3 files)
- **Files**: 
  - `{base_dir}/{exp_name}/data/calibration_set_outputs_generated_data/1/outputs.npz`
  - `{base_dir}/{exp_name}/data/calibration_set_outputs_generated_data/16/outputs.npz`
  - `{base_dir}/{exp_name}/data/calibration_set_outputs_generated_data/32/outputs.npz`
- **Size**: ~50-500 MB each (depends on number of samples)
- **What it contains**: Aggregated predictions from encoder model

## Directory Structure

```
assets/super_resolution/
├── models/
│   └── superres_alpha_0.1.pt                    ← File #1
├── pretrained_models/
│   └── psp_celebs_super_resolution.pt           ← File #2
└── data/
    └── calibration_set_outputs_generated_data/
        ├── 1/
        │   └── outputs.npz                      ← File #3
        ├── 16/
        │   └── outputs.npz                      ← File #4
        └── 32/
            └── outputs.npz                      ← File #5
```

## Do You Need Images?

**Short answer: NO** (if you already have `outputs.npz` files)

**Long answer:**
- ❌ **Images are NOT required** to run the extension experiments if you already have `outputs.npz` files
- ✅ **Images ARE required** only if you need to generate `outputs.npz` files from scratch
- ✅ **For simulation**: You can create synthetic `outputs.npz` files (see below)

## Generating Images and Latents from Scratch

If you don't have `outputs.npz` files, you can generate images and latents using StyleGAN2-FFHQ:

```bash
python extension_experiments/utils/generate_ffhq_data.py \
    --stylegan_path extension_experiments/assets/super_resolution/models/stylegan2-ffhq-config-f.pt \
    --output_base_dir assets/super_resolution/data \
    --num_images 1000 \
    --resize_factors 1,16,32 \
    --device auto  # or 'mps' for Apple Silicon, 'cuda' for NVIDIA GPUs
```

This creates:
- `assets/super_resolution/data/generated/1x/img_00001.png ... img_01000.png`
- `assets/super_resolution/data/generated/1x/latents_00001.npz ... latents_01000.npz`
- Same for `16x/` and `32x/` directories

Then run encoder inference to create `outputs.npz`:
```bash
python extension_experiments/utils/generate_outputs_npz.py \
    --image_dir assets/super_resolution/data/generated/1x \
    --resize_factor 1 \
    --output_path assets/super_resolution/data/calibration_set_outputs_generated_data/1/outputs.npz \
    --batch_size 32  # Reduce to 8 or 4 if you run out of memory (M1 Mac users)
```

## Creating Synthetic Data for Testing

If you're just testing/simulating, you can create fake `outputs.npz` files:

```python
import numpy as np

# Create synthetic outputs.npz
num_samples = 1000
num_dims = 9216  # Style vector dimension

for resize_factor in [1, 16, 32]:
    outputs = {
        'all_sv': np.random.randn(num_samples, num_dims),
        'all_sv_hat': np.random.randn(num_samples, num_dims),
        'all_sv_hat_lq': np.random.randn(num_samples, num_dims) - 0.1,
        'all_sv_hat_uq': np.random.randn(num_samples, num_dims) + 0.1,
        'mean_style_space_vector': np.zeros(num_dims),
        'min_style_space_vector': np.full(num_dims, -2.0),
        'max_style_space_vector': np.full(num_dims, 2.0),
        'style_dims': np.array([
            0, 512, 512, 512, 512, 512, 512, 512, 512, 512,
            512, 512, 512, 512, 512, 512, 256, 256, 256, 128,
            128, 128, 64, 64, 64, 32, 32
        ])
    }
    
    output_path = f'assets/super_resolution/data/calibration_set_outputs_generated_data/{resize_factor}/outputs.npz'
    np.savez(output_path, **outputs)
    print(f"Created {output_path}")
```

## Running Experiments

Once you have the 5 files above, you can run:

```bash
python extension_experiments/run_experiments.py \
    --base_dir assets \
    --exp_name super_resolution \
    --model_name models/superres_alpha_0.1.pt \
    --experiment all
```

## What Happens During Execution

1. **Loads encoder model** from `models/superres_alpha_0.1.pt`
2. **Loads pretrained pSp** from `pretrained_models/psp_celebs_super_resolution.pt`
3. **Loads calibration data** from `outputs.npz` files (3 files)
4. **Computes prediction sets** from the loaded data
5. **Calibrates intervals** (finds optimal lambda values)
6. **Runs experiments** (ablation, stress test, importance ranking)

**No images are loaded during execution** - everything comes from the `outputs.npz` files!

## Summary Table

| File Type | Required? | Count | Purpose |
|-----------|-----------|-------|---------|
| Encoder model (`.pt`) | ✅ Yes | 1 | Predicts style vectors |
| Pretrained pSp (`.pt`) | ✅ Yes | 1 | Generates images from style vectors |
| outputs.npz | ✅ Yes | 3 | Calibration predictions data |
| Image files (`.png`) | ❌ No | 0 | Only needed to generate outputs.npz |
| Latent files (`latents_*.npz`) | ❌ No | 0 | Only needed to generate outputs.npz |

## Next Steps

1. **Get the 5 required files** (or create synthetic outputs.npz for testing)
2. **Run the experiments**: `python extension_experiments/run_experiments.py`
3. **Check results** in `extension_experiments/results/`

For more details, see:
- [INPUT_FILES.md](INPUT_FILES.md) - Complete file specifications
- [DATA_STRUCTURE.md](DATA_STRUCTURE.md) - Directory structure details

