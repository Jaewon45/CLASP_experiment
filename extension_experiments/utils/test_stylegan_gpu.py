import sys
import os
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from argparse import Namespace

# Fix: Use absolute path to CLASP root
script_dir = os.path.dirname(os.path.abspath(__file__))
clasp_root = os.path.abspath(os.path.join(script_dir, "../.."))
sys.path.insert(0, clasp_root)
from pixel2style2pixel.models.stylegan2.model import Generator

@torch.no_grad()
def main():
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load checkpoint exactly as trained
    ckpt_path = "extension_experiments/assets/super_resolution/models/stylegan2-ffhq-config-f.pt"
    print("Loading checkpoint:", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    G = Generator(
        size=1024,
        style_dim=512,
        n_mlp=8,
        channel_multiplier=2
    ).to(device)
    
    G.half()  # saves huge memory
    G.eval()

    print("Loading weights...")
    G.load_state_dict(ckpt["g_ema"], strict=True)
    print("✓ Generator loaded successfully")

    # Test one forward
    print("Running one test generation...")

    z = torch.randn(1, 512, dtype=torch.float16, device=device)
    img, _, _ = G([z], input_is_latent=False, return_latents=True)

    print("✓ Success! Generated image shape:", img.shape)

    # Free memory
    del img, z, G
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
