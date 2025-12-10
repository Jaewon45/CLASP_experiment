# M1/M2/M3 Mac Setup Guide

This guide covers running the extension experiments on Apple Silicon Macs (M1, M2, M3, etc.).

## Important: No CUDA Support

**M1/M2/M3 Macs do NOT support CUDA** (CUDA is NVIDIA-specific). Instead, they use:
- **MPS (Metal Performance Shaders)** - Apple's GPU acceleration framework
- **CPU fallback** - If MPS is not available

## Device Selection

The scripts now support automatic device detection. Use `--device auto` (default) or `--device mps`:

### For M1/M2/M3 Macs:

```bash
# Automatic detection (recommended)
python extension_experiments/utils/generate_ffhq_data.py \
    --stylegan_path extension_experiments/assets/super_resolution/models/stylegan2-ffhq-config-f.pt \
    --output_base_dir assets/super_resolution/data \
    --device auto  # or 'mps' explicitly

# Explicit MPS
python extension_experiments/utils/generate_ffhq_data.py \
    --device mps \
    ...
```

### Device Options:

- `auto` (default): Automatically detects best available device
  - CUDA if available (NVIDIA GPUs)
  - MPS if available (Apple Silicon)
  - CPU otherwise

- `mps`: Force MPS (Apple Silicon GPU)
  - Falls back to CPU if MPS unavailable

- `cpu`: Force CPU (slower but most compatible)

- `cuda`: Force CUDA (will fall back to CPU on Macs)

## PyTorch Requirements

Make sure you have PyTorch with MPS support:

```bash
# Check MPS availability
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

If MPS is not available, install PyTorch with MPS support:

```bash
# For conda
conda install pytorch torchvision torchaudio -c pytorch

# For pip (check latest version)
pip install torch torchvision torchaudio
```

**Note**: MPS support requires PyTorch 1.12+ and macOS 12.3+.

## Performance Considerations

### MPS (Apple Silicon GPU)
- ✅ **Faster** than CPU for most operations
- ✅ Uses unified memory (efficient)
- ⚠️ Some operations may not be supported (falls back to CPU)
- ⚠️ Memory limits depend on system RAM

### CPU
- ✅ Most compatible
- ❌ Slower for large models
- ✅ No memory transfer overhead

## Known Issues & Workarounds

### 1. LPIPS May Use CPU on M1 Macs

**Issue**: The LPIPS implementation in `pixel2style2pixel` hardcodes CUDA. On M1 Macs, it will fall back to CPU.

**Impact**: 
- Ablation experiments (which use LPIPS) will be slower
- Other experiments (stress test, importance ranking) don't use LPIPS, so they're unaffected

**Workaround**: 
- LPIPS will automatically use CPU if CUDA/MPS fails
- This is handled automatically in the code
- Performance will be slower but functional

### 2. Some Operations May Fall Back to CPU

MPS doesn't support all PyTorch operations. If you see warnings like:
```
UserWarning: Operator [operation] is not supported on MPS backend
```

The operation will automatically run on CPU. This is normal and expected.

### 3. Memory Management

MPS uses unified memory. If you run out of memory:

**For `generate_outputs_npz.py` (encoder inference):**
```bash
python extension_experiments/utils/generate_outputs_npz.py \
    --image_dir assets/super_resolution/data/generated/1x \
    --resize_factor 1 \
    --output_path assets/super_resolution/data/calibration_set_outputs_generated_data/1/outputs.npz \
    --batch_size 8  # ← Add this (default is 32)
```

**For `generate_ffhq_data.py` (image generation):**
- This script processes images one at a time, so batch_size doesn't apply
- If memory issues occur, reduce `--num_images`:
```bash
python extension_experiments/utils/generate_ffhq_data.py \
    --num_images 500  # ← Reduce from default 1000
    ...
```

**For experiments:**
- Use `--num_samples` to limit data processing:
```bash
python extension_experiments/run_experiments.py \
    --num_samples 500  # ← Limit samples processed
    ...
```

### 4. StyleGAN2 Custom Operations

Some StyleGAN2 custom CUDA operations may not work on MPS. If you encounter errors:
- The code will fall back to CPU for those operations
- Performance may be slower but should still work

## Example: Full Workflow on M1 Mac

```bash
# Step 1: Generate images and latents (uses MPS automatically)
python extension_experiments/utils/generate_ffhq_data.py \
    --stylegan_path extension_experiments/assets/super_resolution/models/stylegan2-ffhq-config-f.pt \
    --output_base_dir assets/super_resolution/data \
    --num_images 1000 \
    --device auto

# Step 2: Create outputs.npz (uses MPS automatically)
python extension_experiments/utils/generate_outputs_npz.py \
    --image_dir assets/super_resolution/data/generated/1x \
    --resize_factor 1 \
    --output_path assets/super_resolution/data/calibration_set_outputs_generated_data/1/outputs.npz \
    --device auto

# Step 3: Run experiments (uses MPS automatically)
python extension_experiments/run_experiments.py \
    --base_dir assets \
    --exp_name super_resolution \
    --model_name models/superres_alpha_0.1.pt \
    --experiment all
```

## Troubleshooting

### "MPS backend is not available"

**Solution**: 
1. Check PyTorch version: `python -c "import torch; print(torch.__version__)"`
2. Should be 1.12+ for MPS support
3. Update if needed: `pip install --upgrade torch`

### "Out of memory" errors

**Solutions**:

1. **For encoder inference** (`generate_outputs_npz.py`):
   ```bash
   python extension_experiments/utils/generate_outputs_npz.py \
       --batch_size 8 \  # ← Add this flag
       ...
   ```

2. **For image generation** (`generate_ffhq_data.py`):
   ```bash
   python extension_experiments/utils/generate_ffhq_data.py \
       --num_images 500 \  # ← Reduce from 1000
       ...
   ```

3. **For experiments**:
   ```bash
   python extension_experiments/run_experiments.py \
       --num_samples 500 \  # ← Limit samples
       ...
   ```

4. **Last resort**: Use CPU (slower but uses less memory):
   ```bash
   --device cpu
   ```

### Slow performance

**Solutions**:
1. Ensure MPS is being used: Check output for "Using MPS device"
2. Reduce batch size if memory is constrained
3. Close other applications to free up memory

## Quick Command Reference for M1 Mac

```bash
# Generate images (auto-detects MPS)
python extension_experiments/utils/generate_ffhq_data.py \
    --stylegan_path extension_experiments/assets/super_resolution/models/stylegan2-ffhq-config-f.pt \
    --output_base_dir assets/super_resolution/data \
    --device auto  # or 'mps' explicitly

# Create outputs.npz (auto-detects MPS)
python extension_experiments/utils/generate_outputs_npz.py \
    --image_dir assets/super_resolution/data/generated/1x \
    --resize_factor 1 \
    --device auto

# Run experiments (auto-detects MPS)
python extension_experiments/run_experiments.py \
    --base_dir assets \
    --exp_name super_resolution \
    --model_name models/superres_alpha_0.1.pt
```

## Summary

- ✅ **Use `--device auto`** (default) - automatically uses MPS on M1 Macs
- ✅ **MPS is supported** in the updated scripts
- ⚠️ **LPIPS uses CPU** on M1 Macs (pixel2style2pixel limitation)
- ⚠️ **Some operations may fall back to CPU** (this is normal)
- ⚠️ **Memory management** - reduce batch sizes if needed
- ✅ **CPU fallback** - everything will work on CPU if MPS fails

The scripts have been updated to automatically detect and use the best available device on your system.

