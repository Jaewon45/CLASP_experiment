"""
SAFE version of generate_outputs_npz.py
Compatible with:
- StyleGAN2-F 1024px
- StyleGAN2-ADA 256px
- Datasets WITHOUT style vectors
- Datasets missing latents_*.npz
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from argparse import Namespace
from PIL import Image

# -----------------------------
# FIX: Add correct path BEFORE imports
# -----------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
clasp_root = os.path.abspath(os.path.join(script_dir, "../.."))
sys.path.insert(0, clasp_root)
sys.path.insert(0, os.path.join(clasp_root, "pixel2style2pixel"))
# -----------------------------

from pixel2style2pixel.models.encoders import psp_encoders
from pixel2style2pixel.configs import transforms_config
from dataset_utils import RGBSuperResGeneratedDataset
from torch.utils.data import Dataset
import glob


# ------------------------------------------------------------
# Custom dataset that handles PNG files directly in directory
# ------------------------------------------------------------
class DirectImageDataset(Dataset):
    """Dataset that loads PNG files directly from a directory (no subdirectory required)"""
    def __init__(self, db_path, source_transform, target_transform, resolution=256):
        self.db_path = db_path
        # Look for PNG files both directly in directory and in subdirectories
        self.img_fnames = (
            list(glob.glob(os.path.join(self.db_path, '*.png'))) +
            list(glob.glob(os.path.join(self.db_path, '*/*.png')))
        )
        # Sort for consistent ordering
        self.img_fnames = sorted(self.img_fnames)
        self.source_transform = source_transform
        self.target_transform = target_transform
        print(f"Found {len(self.img_fnames)} images in {db_path}")

    def __len__(self):
        return len(self.img_fnames)

    def __getitem__(self, idx):
        img_fname = self.img_fnames[idx]
        rgb_image = Image.open(img_fname)
        input_image = self.source_transform(rgb_image)
        output_image = self.target_transform(rgb_image)
        # Return None for style vectors and wplus since we don't have them
        return input_image, output_image, None, None, img_fname


# ------------------------------------------------------------
# Device selection
# ------------------------------------------------------------
def get_device(device_str="auto"):
    if device_str.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_str)
    if device_str == "cpu":
        return torch.device("cpu")
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device("cpu")


# ------------------------------------------------------------
# SAFE version of outputs generation
# ------------------------------------------------------------
def generate_outputs_npz(
    base_dir,
    exp_name,
    model_name,
    resize_factor,
    image_dir,
    output_path,
    device="auto",
    batch_size=16
):

    device_obj = get_device(device)
    print(f"Using device: {device_obj}")

    # --------------------------------------------------------
    # Load encoder
    # --------------------------------------------------------
    model_path = os.path.join(base_dir, exp_name, model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"Loading encoder from {model_path}")

    ckpt = torch.load(model_path, map_location=device_obj)
    encoder_opts = ckpt["opts"]
    encoder_opts["checkpoint_path"] = model_path
    encoder_opts = Namespace(**encoder_opts)

    encoder = psp_encoders.SimpleResnetEncoder_with_quantiles(
        num_layers=50,
        mode='ir',
        style_dims=9088,
        opts=encoder_opts
    )   

    # Strip 'encoder.' prefix from checkpoint keys and map to model's expected names
    # Only process keys that start with 'encoder.' and ignore all others
    ckpt_filt = {}
    encoder_key_count = 0
    for k, v in ckpt['state_dict'].items():
        if k.startswith('encoder.'):
            # Remove 'encoder.' prefix to match model's expected key names
            new_key = k[len('encoder.'):]
            
            # Filter out keys that the model doesn't expect
            if new_key.startswith('latlayer'):
                # Skip latlayer keys (not in SimpleResnetEncoder_with_quantiles)
                continue
            if '.res_layer.5.' in new_key:
                # Skip extra fc layers in resnet blocks (not in this model variant)
                continue
            
            # Map 'styles.*' to 'pooling_block.*' to match model architecture
            if new_key.startswith('styles.'):
                new_key = new_key.replace('styles.', 'pooling_block.', 1)
            
            ckpt_filt[new_key] = v
            encoder_key_count += 1
        # Ignore all other keys (decoder, etc.)
    
    print(f"Found {encoder_key_count} encoder keys in checkpoint (out of {len(ckpt['state_dict'])} total keys)")
    print(f"Filtered to {len(ckpt_filt)} keys for model")
    
    # Use strict=False to allow missing projection layers (they may be initialized randomly)
    encoder.load_state_dict(ckpt_filt, strict=False)
    encoder.eval().to(device_obj)

    print("✓ Encoder loaded successfully.")

    # --------------------------------------------------------
    # Transforms & dataset
    # --------------------------------------------------------
    encoder_opts.resize_factors = str(resize_factor)
    transform_obj = transforms_config.SuperResTransforms(encoder_opts).get_transforms()

    # Use DirectImageDataset to handle PNG files directly in the directory
    dataset = DirectImageDataset(
        db_path=image_dir,
        source_transform=transform_obj["transform_source"],
        target_transform=transform_obj["transform_gt_train"],
        resolution=256,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f"Dataset size: {len(dataset)} images")
    
    # Check if dataset is empty
    if len(dataset) == 0:
        print(f"⚠ ERROR: Dataset is empty!")
        print(f"   Image directory: {image_dir}")
        print(f"   Directory exists: {os.path.exists(image_dir)}")
        if os.path.exists(image_dir):
            # List contents of directory
            try:
                files = os.listdir(image_dir)
                print(f"   Files in directory: {len(files)}")
                if len(files) > 0:
                    print(f"   First 5 files: {files[:5]}")
            except Exception as e:
                print(f"   Error listing directory: {e}")
        raise ValueError(f"Dataset is empty. No images found in {image_dir}")

    # --------------------------------------------------------
    # Output arrays
    # --------------------------------------------------------
    all_sv = []         # ground truth style vectors  (may be empty or dummy)
    all_sv_hat = []     # predicted mean style vectors
    all_sv_hat_lq = []  # predicted lq
    all_sv_hat_uq = []  # predicted uq

    print("Running inference...")

    # --------------------------------------------------------
    # Loop over batches
    # --------------------------------------------------------
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing"):

            # Unpack batch
            input_image, output_image, true_style_vectors, true_wplus, img_names = batch

            input_image = input_image.to(device_obj).float()

            # ------------------------------------------------
            # Forward pass
            # ------------------------------------------------
            preds = encoder(input_image)

            # Expected output shape: [B, 3, D]
            preds = preds.cpu().numpy()

            # Quantiles
            lq = preds[:, 0, :]
            mean = preds[:, 1, :]
            uq = preds[:, 2, :]

            all_sv_hat.append(mean)
            all_sv_hat_lq.append(lq)
            all_sv_hat_uq.append(uq)

            # ------------------------------------------------
            # SAFE handling of ground truth vectors (optional)
            # ------------------------------------------------
            if isinstance(true_style_vectors, torch.Tensor):
                # If provided and valid, use it. Otherwise skip.
                for sv in true_style_vectors:
                    if sv is None or torch.isnan(sv).any():
                        continue
                    sv = sv.cpu().numpy().flatten()
                    all_sv.append(sv)

    # Stack predictions - handle empty case
    if len(all_sv_hat) == 0:
        raise ValueError("No predictions generated. All batches were empty.")
    
    all_sv_hat = np.vstack(all_sv_hat)
    all_sv_hat_lq = np.vstack(all_sv_hat_lq)
    all_sv_hat_uq = np.vstack(all_sv_hat_uq)

    # --------------------------------------------------------
    # Handle missing GT style vectors
    # --------------------------------------------------------
    if len(all_sv) == 0:
        print("⚠ No ground-truth style vectors found — creating dummy array.")
        gt_dim = all_sv_hat.shape[1]
        all_sv = np.zeros((len(all_sv_hat), gt_dim))
    else:
        all_sv = np.vstack(all_sv)

    # --------------------------------------------------------
    # Infer semantic dimension size
    # --------------------------------------------------------
    style_dims = all_sv_hat.shape[1]

    # --------------------------------------------------------
    # Save everything
    # --------------------------------------------------------
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    np.savez(
        output_path,
        all_sv=all_sv,
        all_sv_hat=all_sv_hat,
        all_sv_hat_lq=all_sv_hat_lq,
        all_sv_hat_uq=all_sv_hat_uq,
        style_dims=style_dims,
    )

    print(f"\n✓ Saved outputs.npz to {output_path}")
    print(f"  - all_sv:         {all_sv.shape}")
    print(f"  - all_sv_hat:     {all_sv_hat.shape}")
    print(f"  - all_sv_hat_lq:  {all_sv_hat_lq.shape}")
    print(f"  - all_sv_hat_uq:  {all_sv_hat_uq.shape}")
    print(f"  - style_dims:     {style_dims}")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default="assets")
    parser.add_argument("--exp_name", default="super_resolution")
    parser.add_argument("--model_name", default="models/superres_alpha_0.1.pt")
    parser.add_argument("--resize_factor", type=int, required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="auto")

    args = parser.parse_args()

    generate_outputs_npz(
        base_dir=args.base_dir,
        exp_name=args.exp_name,
        model_name=args.model_name,
        resize_factor=args.resize_factor,
        image_dir=args.image_dir,
        output_path=args.output_path,
        device=args.device,
        batch_size=args.batch_size,
    )
