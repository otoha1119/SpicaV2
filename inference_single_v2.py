"""Tile-based inference script matching training normalization.

This script reads a single DICOM file, normalizes the pixel values with
(arr + 2048) / 6143, runs the generator patch-by-patch with 50% overlap, and
blends the 2x super-resolved patches using a cosine window.
"""
import math
import sys
from typing import Tuple

import numpy as np
import torch

from options.test_options import TestOptions
from models import create_model
from util.dicom_io import read_normalized_pixels, save_dicom_like


PATCH_SIZE = 98
STRIDE = 49
SCALE = 2
PATCH_OUT = PATCH_SIZE * SCALE  # 196


def _enforce_dataset_mode():
    """Force the dataset mode to dicom_ctpcct_2x_test before parsing options."""
    argv = sys.argv
    if "--dataset_mode" not in argv:
        argv.extend(["--dataset_mode", "dicom_ctpcct_2x_test"])


def parse_options():
    """Parse inference options using TestOptions with enforced settings."""
    _enforce_dataset_mode()
    opt = TestOptions().parse()

    # Single-image inference specific settings
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.dataset_mode = "dicom_ctpcct_2x_test"

    return opt


def build_cosine_weight(patch_out: int = PATCH_OUT) -> np.ndarray:
    """Create a 2D cosine window for blending patches."""
    x = np.arange(patch_out, dtype=np.float32)
    wx = 0.5 * (1.0 - np.cos(2.0 * math.pi * x / (patch_out - 1)))
    weight = np.outer(wx, wx)
    return weight.astype(np.float32)


def compute_padding(length: int, patch: int, stride: int) -> Tuple[int, int, int]:
    """Compute symmetric reflect padding to fully cover the dimension.

    Returns (pad_before, pad_after, padded_length).
    """
    steps = math.ceil((length - patch) / stride) + 1
    total = patch + stride * (steps - 1)
    pad_needed = max(0, total - length)
    pad_before = pad_needed // 2
    pad_after = pad_needed - pad_before
    padded_len = length + pad_needed
    return pad_before, pad_after, padded_len


def pad_reflect(img: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Pad the image with reflection so that sliding windows cover the edges."""
    h, w = img.shape
    pt, pb, _ = compute_padding(h, PATCH_SIZE, STRIDE)
    pl, pr, _ = compute_padding(w, PATCH_SIZE, STRIDE)
    if any(v > 0 for v in (pt, pb, pl, pr)):
        img = np.pad(img, ((pt, pb), (pl, pr)), mode="reflect")
    return img, (pl, pr, pt, pb)


def select_generator(model, use_g: str = "A"):
    """Return the configured generator (netG_A or netG_B)."""
    if use_g.upper() == "B":
        return model.netG_B
    return model.netG_A


def run_tiled_inference(model, lr_img: np.ndarray, device: torch.device, use_g: str = "A") -> np.ndarray:
    """Run tiled super-resolution on a single normalized image."""
    padded, (pl, pr, pt, pb) = pad_reflect(lr_img)
    ph, pw = padded.shape
    sr_h = ph * SCALE
    sr_w = pw * SCALE

    weight = build_cosine_weight(PATCH_OUT)
    accumulator = np.zeros((sr_h, sr_w), dtype=np.float32)
    weight_sum = np.zeros_like(accumulator)

    generator = select_generator(model, use_g)
    generator.eval()

    with torch.no_grad():
        for y in range(0, ph - PATCH_SIZE + 1, STRIDE):
            for x in range(0, pw - PATCH_SIZE + 1, STRIDE):
                patch = padded[y : y + PATCH_SIZE, x : x + PATCH_SIZE]
                inp = torch.from_numpy(patch)[None, None, ...].to(device)
                sr_tensor = generator(inp)
                sr_patch = sr_tensor[0, 0].detach().cpu().numpy()

                sy = y * SCALE
                sx = x * SCALE
                accumulator[sy : sy + PATCH_OUT, sx : sx + PATCH_OUT] += sr_patch * weight
                weight_sum[sy : sy + PATCH_OUT, sx : sx + PATCH_OUT] += weight

    final_padded = accumulator / np.maximum(weight_sum, 1e-8)
    final_padded = np.clip(final_padded, 0.0, 1.0)

    # Remove the scaled padding to return to the original 2x size
    crop_top = pt * SCALE
    crop_left = pl * SCALE
    target_h = lr_img.shape[0] * SCALE
    target_w = lr_img.shape[1] * SCALE
    final_sr = final_padded[crop_top : crop_top + target_h, crop_left : crop_left + target_w]
    return final_sr


def main():
    opt = parse_options()

    # Load and normalize input with the exact training rule
    lr_norm = read_normalized_pixels(opt.input_dicom)
    if lr_norm.ndim == 3:
        lr_norm = lr_norm[0]

    device = torch.device(f"cuda:{opt.gpu_ids[0]}" if getattr(opt, "gpu_ids", None) else "cpu")

    # Create and load the model
    model = create_model(opt)
    model.setup(opt)
    model.eval()

    sr_img = run_tiled_inference(model, lr_norm, device, getattr(opt, "use_G", "A"))

    # Save as DICOM with the provided helper
    save_dicom_like(opt.input_dicom, opt.output_dicom, sr_img)
    print(f"Saved super-resolved image to {opt.output_dicom}")


if __name__ == "__main__":
    main()
