# Error Handling and Fallbacks

This document details where error handling and fallbacks are implemented in the extension framework.

## 1. Import Fallbacks

### `corruptions/corruption_utils.py`
**Location**: Lines 19-24, 43-48

```python
# Try to import cv2, fallback to torch-based blur
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# In apply_blur():
if HAS_CV2:
    kernel = cv2.getGaussianKernel(kernel_size, sigma)
    # ... use cv2
else:
    # Fallback: create Gaussian kernel using torch
    coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    # ... use torch-based implementation
```

**Purpose**: Allows the code to work even if OpenCV is not installed.

### `utils/lpips_utils.py`
**Location**: Lines 12-18, 33-35

```python
try:
    from pixel2style2pixel.criteria.lpips.lpips import LPIPS
    HAS_LPIPS = True
except (ImportError, ModuleNotFoundError):
    LPIPS = None
    HAS_LPIPS = False

# In __init__:
if not HAS_LPIPS or LPIPS is None:
    raise ImportError("LPIPS module not found...")
```

**Purpose**: Gracefully handles missing LPIPS module with clear error message.

### `utils/dataset_utils.py`
**Location**: Lines 17-21

```python
try:
    from extension_experiments.corruptions.corruption_utils import apply_corruption_level
except ImportError:
    # Fallback for direct import
    from corruptions.corruption_utils import apply_corruption_level
```

**Purpose**: Handles different import path scenarios.

## 2. Exception Handling in Experiments

### `experiments/ablation_experiment.py`
**Location**: Lines 167-193

```python
for pert_type in perturbation_types:
    try:
        stability, low_im, high_im = self.compute_stability_for_dim(...)
        dim_results[pert_type] = {
            'stability_score': float(stability),
            'dimension_index': int(actual_dim)
        }
        # ... save visualization
    except Exception as e:
        print(f"Error processing dim {actual_dim} with {pert_type}: {e}")
        dim_results[pert_type] = {'stability_score': None, 'error': str(e)}
```

**Purpose**: Continues processing other dimensions/perturbations even if one fails.

### `experiments/stress_test.py`
**Location**: Lines 118-147

```python
for corruption_level in tqdm(self.corruption_levels, ...):
    try:
        interval_widths, lower_bounds, upper_bounds = self.compute_intervals_for_corruption(...)
        # ... process results
    except Exception as e:
        print(f"Error processing {corruption_level}: {e}")
        results_dict['corruption_levels'][corruption_level] = {'error': str(e)}
```

**Purpose**: Continues with other corruption levels if one fails.

### `experiments/importance_ranking.py`
**Location**: Lines 187-195

```python
for difficulty_level in tqdm(difficulty_levels, ...):
    try:
        ranking_df = self.rank_dimensions(...)
        comparison_dict[difficulty_level] = ranking_df.to_dict('records')
    except Exception as e:
        print(f"Error processing {difficulty_level}: {e}")
        comparison_dict[difficulty_level] = {'error': str(e)}
```

**Purpose**: Handles errors when comparing across difficulty levels.

### `example_usage.py`
**Location**: Lines 124-138

```python
try:
    lpips = LPIPSMetric(device='cuda' if torch.cuda.is_available() else 'cpu')
    distance = lpips.compute(img1, img2)
    print(f"LPIPS distance: {distance:.4f}")
except ImportError as e:
    print(f"\nLPIPS not available: {e}")
    print("This is expected if pixel2style2pixel is not fully set up.")
```

**Purpose**: Example code that gracefully handles missing dependencies.

## 3. Runtime Checks

### `run_experiments.py`
**Location**: Lines 47-50

```python
# Compute losses and calibrate if needed
if not hasattr(results_runner, 'all_rcps_stats') or len(results_runner.all_rcps_stats['easy']) == 0:
    results_runner.compute_losses_prediction_sets()
    results_runner.calibrate_all_difficulty_levels(total_runs=100)
```

**Purpose**: Automatically computes calibration if not already done.

### `utils/dataset_utils.py`
**Location**: Lines 49-50

```python
if filelist is None:
    # Look for images in standard structure
```

**Purpose**: Handles both filelist and automatic discovery scenarios.

## 4. Data Validation

### `utils/dataset_utils.py`
**Location**: Lines 60-61

```python
if os.path.exists(latent_fname):
    # Load latents
else:
    # Latents are optional
    true_style_vectors = None
    true_wplus = None
```

**Purpose**: Handles missing latent files gracefully.

## Summary

Error handling is implemented in:
1. **Import fallbacks**: cv2, LPIPS, module imports
2. **Experiment loops**: Try/except blocks to continue processing
3. **Runtime checks**: Automatic calibration if needed
4. **Data validation**: Optional file existence checks
5. **Example code**: Graceful degradation in demos

All error handling follows the principle of "fail gracefully, continue processing" where possible, and provides clear error messages when critical components are missing.

