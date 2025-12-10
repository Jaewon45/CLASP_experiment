# Data Structure and Storage Requirements

This document explains where data should be stored and the expected directory structure.

## Expected Directory Structure

Based on the `RCPS_Results_Runner` implementation, the expected structure is:

```
{base_dir}/
└── {exp_name}/                    # e.g., 'super_resolution'
    ├── models/
    │   └── {model_name}           # e.g., 'superres_alpha_0.1.pt'
    ├── pretrained_models/
    │   └── psp_celebs_super_resolution.pt
    └── data/
        └── calibration_set_outputs_generated_data/
            ├── {resize_factor_1}/  # e.g., '1', '16', '32'
            │   └── outputs.npz
            ├── {resize_factor_2}/
            │   └── outputs.npz
            └── {resize_factor_3}/
                └── outputs.npz
```

## 1. FFHQ Data with Latent Vectors

### Storage Location

The FFHQ data generated via StyleGAN should be stored with the following structure:

**Option A: Using existing calibration data structure**
```
{base_dir}/{exp_name}/data/calibration_set_outputs_generated_data/{resize_factor}/
├── img_*.png              # Generated images
└── latents_*.npz          # Corresponding latent vectors
```

**Option B: Separate dataset directory (for new data)**
```
{base_dir}/{exp_name}/data/ffhq_generated/
├── {subfolder_1}/
│   ├── img_00001.png
│   ├── latents_00001.npz
│   ├── img_00002.png
│   └── latents_00002.npz
└── {subfolder_2}/
    └── ...
```

### Latent Vector Format (.npz files)

Each `.npz` file should contain:
```python
{
    'style_vectors': np.ndarray,  # Style space vectors
    'wplus': np.ndarray,          # W+ latent codes
    # Optional:
    'mean_style_space_vector': np.ndarray,
    'min_style_space_vector': np.ndarray,
    'max_style_space_vector': np.ndarray,
    'style_dims': np.ndarray
}
```

### Example: Loading Latent Data

The `FFHQCorruptedDataset` class expects this structure:
```python
latent_fname = img_path.replace('/img_', '/latents_').replace('.png', '.npz')
with np.load(latent_fname) as data:
    true_style_vectors = torch.Tensor(data['style_vectors'])
    true_wplus = torch.Tensor(data['wplus'])
```

## 2. Calibrated Prediction Sets

### Storage Location

Calibrated prediction sets are **computed at runtime** and stored in memory. They are **not** saved to disk by default, but are computed from:

**Input**: `{base_dir}/{exp_name}/data/calibration_set_outputs_generated_data/{resize_factor}/outputs.npz`

This file should contain:
```python
{
    'all_sv': np.ndarray,              # All style vectors (ground truth)
    'all_sv_hat': np.ndarray,           # Predicted style vectors
    'all_sv_hat_lq': np.ndarray,        # Lower quantile predictions
    'all_sv_hat_uq': np.ndarray,        # Upper quantile predictions
    'mean_style_space_vector': np.ndarray,
    'min_style_space_vector': np.ndarray,
    'max_style_space_vector': np.ndarray,
    'style_dims': np.ndarray
}
```

### Calibration Process

Calibration happens in `RCPS_Results_Runner`:

1. **Load data**: `load_data()` reads from `outputs.npz` files
2. **Compute prediction sets**: `compute_losses_prediction_sets()` creates nested sets
3. **Calibrate**: `calibrate_all_difficulty_levels()` finds optimal lambda values
4. **Store in memory**: Results stored in `all_rcps_stats`, `all_prediction_sets`, etc.

### Saving Calibration Results (Optional)

To save calibration results for later use, you could add:

```python
# After calibration
import pickle
calibration_save_path = f"{base_dir}/{exp_name}/calibration_results.pkl"
with open(calibration_save_path, 'wb') as f:
    pickle.dump({
        'all_rcps_stats': results_runner.all_rcps_stats,
        'all_prediction_sets': results_runner.all_prediction_sets,
        'all_losses_per_lambda': results_runner.all_losses_per_lambda,
        'test_idx_list': results_runner.test_idx_list
    }, f)
```

## 3. Default Paths

Based on the code in `qgan_superres_results_runner.py`:

```python
# Default values (from notebooks/superres_results_wrapper.py)
base_dir = 'assets'
exp_name = 'super_resolution'
model_name = 'models/superres_alpha_0.1.pt'
resize_factors = [1, 16, 32]
difficulty_levels = ['easy', 'medium', 'hard']
```

This translates to:
- **Model**: `assets/super_resolution/models/superres_alpha_0.1.pt`
- **Calibration data**: `assets/super_resolution/data/calibration_set_outputs_generated_data/{1,16,32}/outputs.npz`
- **Pretrained models**: `assets/super_resolution/pretrained_models/psp_celebs_super_resolution.pt`

## 4. Creating the Data Structure

### Step 1: Generate FFHQ Data with StyleGAN

Generate images and save latents:
```python
# Pseudocode
for i in range(num_images):
    # Generate image and latents using StyleGAN
    img, style_vectors, wplus = generate_stylegan_image(...)
    
    # Save image
    img_path = f"{base_dir}/{exp_name}/data/ffhq_generated/img_{i:05d}.png"
    Image.fromarray(img).save(img_path)
    
    # Save latents
    latent_path = img_path.replace('/img_', '/latents_').replace('.png', '.npz')
    np.savez(latent_path, 
             style_vectors=style_vectors,
             wplus=wplus)
```

### Step 2: Create Calibration Outputs

Run inference on generated data to create `outputs.npz`:

**Option A: Use the provided script**

```bash
python extension_experiments/utils/generate_outputs_npz.py \
    --base_dir assets \
    --exp_name super_resolution \
    --model_name models/superres_alpha_0.1.pt \
    --resize_factor 1 \
    --image_dir assets/super_resolution/data/generated/1x \
    --output_path assets/super_resolution/data/calibration_set_outputs_generated_data/1/outputs.npz \
    --batch_size 32  # Reduce to 8 or 4 if you run out of memory
```

Repeat for each resize factor (1, 16, 32, etc.).

**Option B: Manual generation**

The `outputs.npz` file contains predictions from the encoder model. You need to:
1. Load your trained encoder model
2. Run inference on all images in your dataset
3. Collect predictions: `all_sv_hat`, `all_sv_hat_lq`, `all_sv_hat_uq`
4. Save along with ground truth `all_sv` and statistics

See `extension_experiments/utils/generate_outputs_npz.py` for reference implementation.

### Step 3: Run Experiments

The extension experiments will automatically:
1. Load calibration data from `outputs.npz`
2. Compute prediction sets
3. Calibrate intervals
4. Run experiments

## 5. Custom Data Paths

To use custom paths, modify the arguments:

```python
results_runner = RCPS_Results_Runner(
    base_dir='path/to/your/data',      # Custom base directory
    exp_name='your_experiment',        # Custom experiment name
    model_name='your_model.pt',        # Custom model path
    resize_factors=[1, 16, 32],
    difficulty_levels=['easy', 'medium', 'hard'],
    norm_scheme='mean'
)
```

The code will automatically construct paths:
- `{base_dir}/{exp_name}/data/calibration_set_outputs_generated_data/{resize_factor}/outputs.npz`

## Summary

- **FFHQ Data**: Store in `{base_dir}/{exp_name}/data/` with images and corresponding `.npz` latent files
- **Calibration Data**: Store in `{base_dir}/{exp_name}/data/calibration_set_outputs_generated_data/{resize_factor}/outputs.npz`
- **Calibrated Sets**: Computed at runtime, stored in memory (can be saved with pickle if needed)
- **Default**: `assets/super_resolution/` structure

## Important: Generating .npz Files

**The code does NOT automatically generate `.npz` files from images.** You need to generate them separately:

1. **Individual `latents_*.npz` files**: These should be created when generating images with StyleGAN. Each image should have a corresponding latent file.

2. **`outputs.npz` files**: These are created by running the encoder model on your images. Use the provided script:
   ```bash
   python extension_experiments/utils/generate_outputs_npz.py \
       --resize_factor 1 \
       --image_dir <your_image_directory> \
       --output_path <output_path>/outputs.npz
   ```

The extension experiments expect these files to already exist before running.

