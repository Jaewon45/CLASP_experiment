# Memory Management: Using `num_samples` Parameter

This document explains how to adjust memory usage using the `num_samples` parameter and how it works.

## What is `num_samples`?

The `num_samples` parameter limits the number of data samples processed when computing statistics. This is useful for:
- **Reducing GPU memory usage** when working with large datasets
- **Faster experimentation** during development
- **Testing** with smaller subsets before running full experiments

## How It Works

### Default Behavior

When `num_samples=None` (default):
- **All available samples** are processed
- Uses the full dataset from calibration data
- Maximum accuracy but highest memory usage

### With `num_samples` Set

When `num_samples=N` (e.g., `num_samples=1000`):
- Only the **first N samples** are processed
- Reduces memory footprint proportionally
- Faster computation
- Still statistically meaningful for large N

### Implementation Details

The parameter works by **slicing the data arrays** before computation:

```python
# In stress_test.py and importance_ranking.py
lower_edges_selected = lower_edges[:, self.disent_dims[self.select_indices]]
upper_edges_selected = upper_edges[:, self.disent_dims[self.select_indices]]

# Limit samples if requested
if num_samples is not None:
    lower_edges_selected = lower_edges_selected[:num_samples]  # First N samples
    upper_edges_selected = upper_edges_selected[:num_samples]  # First N samples
```

This happens **before** computing statistics, so memory usage is reduced from the start.

## Where to Use It

### 1. Stress Test Experiment

**Command line:**
```bash
python extension_experiments/experiments/stress_test.py \
    --num_samples 1000 \
    --base_dir assets \
    --exp_name super_resolution \
    --model_name models/superres_alpha_0.1.pt
```

**In code:**
```python
stress_test = StressTestExperiment(results_runner)
results = stress_test.run_stress_test(
    run_index=-1,
    num_samples=1000,  # Process only first 1000 samples
    output_dir='results/stress_test'
)
```

### 2. Importance Ranking Experiment

**Command line:**
```bash
python extension_experiments/experiments/importance_ranking.py \
    --num_samples 500 \
    --base_dir assets \
    --exp_name super_resolution \
    --model_name models/superres_alpha_0.1.pt
```

**In code:**
```python
ranking_exp = ImportanceRankingExperiment(results_runner)
ranking = ranking_exp.rank_dimensions(
    difficulty_level='easy',
    run_index=-1,
    num_samples=500,  # Process only first 500 samples
    output_dir='results/importance_ranking'
)
```

### 3. Main Runner Script

**Command line:**
```bash
python extension_experiments/run_experiments.py \
    --experiment all \
    --num_samples 1000 \
    --base_dir assets \
    --exp_name super_resolution
```

This applies `num_samples` to both stress test and importance ranking experiments.

## Memory Impact

### Without `num_samples` (Full Dataset)

Assuming:
- Dataset size: 10,000 samples
- Dimensions: 33 (disentangled dimensions)
- Data type: float32 (4 bytes)

**Memory for bounds arrays:**
- Lower bounds: 10,000 × 33 × 4 bytes = ~1.3 MB
- Upper bounds: 10,000 × 33 × 4 bytes = ~1.3 MB
- **Total**: ~2.6 MB (just for bounds)

**Additional memory** for:
- Intermediate computations
- LPIPS metric (if used)
- Image tensors (if generating visualizations)

### With `num_samples=1000`

**Memory for bounds arrays:**
- Lower bounds: 1,000 × 33 × 4 bytes = ~0.13 MB
- Upper bounds: 1,000 × 33 × 4 bytes = ~0.13 MB
- **Total**: ~0.26 MB (10× reduction)

### With `num_samples=100`

**Memory for bounds arrays:**
- Lower bounds: 100 × 33 × 4 bytes = ~0.013 MB
- Upper bounds: 100 × 33 × 4 bytes = ~0.013 MB
- **Total**: ~0.026 MB (100× reduction)

## Choosing the Right Value

### For Development/Testing
```python
num_samples = 100  # Quick tests, minimal memory
```

### For Medium-Scale Experiments
```python
num_samples = 1000  # Good balance of speed and accuracy
```

### For Full Results
```python
num_samples = None  # Use all data, maximum accuracy
```

### Statistical Considerations

- **Minimum**: 100-200 samples for basic statistics
- **Recommended**: 1000+ samples for reliable estimates
- **Full dataset**: Use `None` for publication-quality results

The interval width computation (`W_d = E[q_d^high - q_d^low]`) is an **average**, so:
- With 1000 samples: Standard error is typically small enough
- With 100 samples: May have higher variance but still informative
- With full dataset: Maximum statistical power

## Example: Progressive Testing

```python
# Step 1: Quick test with small sample
results_quick = stress_test.run_stress_test(num_samples=100)

# Step 2: Medium-scale test
results_medium = stress_test.run_stress_test(num_samples=1000)

# Step 3: Full experiment (if memory allows)
results_full = stress_test.run_stress_test(num_samples=None)
```

## Monitoring Memory Usage

### Check GPU Memory (if using CUDA)
```python
import torch
print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"GPU Memory Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

### Check System Memory (Python)
```python
import psutil
process = psutil.Process()
print(f"Memory Usage: {process.memory_info().rss / 1e9:.2f} GB")
```

## Troubleshooting

### Out of Memory Error

**Solution 1**: Reduce `num_samples`
```python
num_samples = 500  # Instead of None or 10000
```

**Solution 2**: Process in batches
```python
# Not currently implemented, but you could modify the code to:
for batch_start in range(0, total_samples, batch_size):
    batch_end = min(batch_start + batch_size, total_samples)
    # Process batch
```

**Solution 3**: Use CPU instead of GPU
```python
device = 'cpu'  # Slower but uses system RAM instead of GPU memory
```

### Slow Performance

If `num_samples=None` is too slow:
- Start with `num_samples=1000` to verify the pipeline works
- Gradually increase to find the sweet spot
- Use `num_samples=None` only for final results

## Summary

- **`num_samples=None`**: Process all samples (default, maximum memory)
- **`num_samples=N`**: Process first N samples (reduced memory, faster)
- **Recommended**: Start with 1000 for testing, use `None` for final results
- **Memory savings**: Linear reduction (1000 samples = 10× less memory than 10000)

The parameter is available in:
- `stress_test.py` → `run_stress_test(num_samples=...)`
- `importance_ranking.py` → `rank_dimensions(num_samples=...)`
- `run_experiments.py` → `--num_samples` command-line argument

