import sys
import os
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# Fix module paths
script_dir = os.path.dirname(os.path.abspath(__file__))
clasp_root = os.path.abspath(os.path.join(script_dir, "../.."))
sys.path.insert(0, clasp_root)

from pixel2style2pixel.models.stylegan2.model import Generator


# ------------------------------------------------------------------------------
# Utility: Convert tensor → image safely
# ------------------------------------------------------------------------------
def tensor2im(tensor):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu()

    # Remove batch dim
    if tensor.ndim == 4:
        tensor = tensor[0]

    # (C,H,W)
    if tensor.ndim != 3:
        raise ValueError(f"tensor2im received invalid shape {tensor.shape}")

    # Single channel → 3 channels
    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)

    # (C,H,W) → (H,W,C)
    arr = tensor.permute(1, 2, 0).numpy()

    # Convert from [-1,1] to uint8
    if arr.min() < 0:
        arr = (arr + 1) / 2
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)

    return Image.fromarray(arr)


# ------------------------------------------------------------------------------
# Load StyleGAN2 (handles both 1024-F and 256-ADA)
# ------------------------------------------------------------------------------
def load_stylegan_safe(path, device):
    print(f"Loading checkpoint: {path}")

    # torch.load MUST use weights_only=False for .pkl
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")  # older pytorch

    # Determine resolution
    if "g_ema" in ckpt:
        # pSp / StyleGAN2-F format
        print("Detected: StyleGAN2-F 1024px")
        size = 1024
        G = Generator(size=1024, style_dim=512, n_mlp=8, channel_multiplier=2)

        G.load_state_dict(ckpt["g_ema"], strict=False)
        latent_avg = ckpt.get("latent_avg", None)

    elif "G_ema" in ckpt:
        # ADA format (1024 or 256)
        print("Detected: StyleGAN2-ADA model")

        # Infer resolution
        try:
            size = ckpt["G_ema"].img_resolution
        except:
            size = 256

        G = Generator(size=size, style_dim=512, n_mlp=8, channel_multiplier=2)
        G.load_state_dict(ckpt["G_ema"].state_dict(), strict=False)
        latent_avg = torch.zeros(1, 512)  # ADA pkls often don't include latent_avg

    else:
        raise ValueError("Unknown checkpoint format.")

    G = G.to(device).eval().half()

    print(f"Loaded StyleGAN generator @ resolution {size}px")
    return G, latent_avg, size


# ------------------------------------------------------------------------------
# SAFE main generator
# ------------------------------------------------------------------------------
def generate_ffhq_data(
    stylegan_path,
    output_base_dir,
    device="cuda",
    num_images=10,
    batch_size=1,   # forced
    resize_factors=[1],
    compute_stats=True
):

    # Device selection
    if device.startswith("cuda") and torch.cuda.is_available():
        device_obj = torch.device(device)
    else:
        device_obj = torch.device("cpu")
    print("Forced device:", device_obj)

    # Load generator safely
    G, latent_avg, size = load_stylegan_safe(stylegan_path, device_obj)

    # Create output dirs
    output_base = Path(output_base_dir)
    for r in resize_factors:
        (output_base / f"generated/{r}x").mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating {num_images} images...\n")

    # Style stats storage (only if compatible)
    style_vec_len = 18 * 512          # expected for StyleGAN2-F
    style_stats_enabled = True

    sum_vec = np.zeros(style_vec_len, dtype=np.float64)
    min_vec = np.full(style_vec_len, +np.inf)
    max_vec = np.full(style_vec_len, -np.inf)
    count = 0

    # Generation loop
    for i in tqdm(range(num_images), desc="Generating"):
        with torch.cuda.amp.autocast(dtype=torch.float16):
            z = torch.randn(1, 512, device=device_obj)
            img, style_vectors, wplus = G([z], input_is_latent=False, return_latents=True)

        # -------------------------------------------------------------
        # Save image
        # -------------------------------------------------------------
        img_pil = tensor2im(img)
        for r in resize_factors:
            resized = img_pil.resize((size // r, size // r))
            out_path = output_base / f"generated/{r}x/{i:05d}.png"
            resized.save(out_path)

        # -------------------------------------------------------------
        # Attempt style vector accumulation (safe)
        # -------------------------------------------------------------
        if compute_stats and style_stats_enabled:
            try:
                sv = style_vectors.detach().cpu().numpy().reshape(-1)
                if sv.shape[0] != style_vec_len:
                    print(f"[WARNING] Incompatible style vector shape {sv.shape}, skipping stats.")
                    style_stats_enabled = False
                else:
                    sum_vec += sv
                    min_vec = np.minimum(min_vec, sv)
                    max_vec = np.maximum(max_vec, sv)
                    count += 1
            except:
                print("[WARNING] Style vectors not compatible. Stats disabled.")
                style_stats_enabled = False

        del img, style_vectors, wplus, z
        torch.cuda.empty_cache()

    # -------------------------------------------------------------
    # Save stats (only if compatible)
    # -------------------------------------------------------------
    if compute_stats and style_stats_enabled and count > 0:
        mean_vec = sum_vec / count
        np.save(output_base / "mean_style_space_vector.npy", mean_vec)
        np.save(output_base / "min_style_space_vector.npy", min_vec)
        np.save(output_base / "max_style_space_vector.npy", max_vec)
        print("\nSaved style-space statistics.")
    else:
        print("\nNo valid style vectors. Stats skipped.")

    return


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--stylegan_path", required=True)
    parser.add_argument("--output_base_dir", default="assets/data")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_images", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--resize_factors", nargs="+", type=int, default=[1])
    parser.add_argument("--no_stats", action="store_true")

    args = parser.parse_args()

    generate_ffhq_data(
        stylegan_path=args.stylegan_path,
        output_base_dir=args.output_base_dir,
        device=args.device,
        num_images=args.num_images,
        batch_size=1,
        resize_factors=args.resize_factors,
        compute_stats=not args.no_stats
    )
