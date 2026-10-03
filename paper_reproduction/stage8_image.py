"""
stage8_image.py -- Stage 7: STFT #2 and 128x128 Continuous Grayscale Image Construction.

Authoritative Reference:
  PREPROCESSING_PIPELINE.md (Section 1-3, D10, D12, D13, D14, D15, D16)
  config_paper_test.yaml (blocks: stft2, image, h1)

Lineage Categories:
  - 128x128, continuous [0,1], NEVER binarized: [PAPER-SPECIFIED] (Q6).
  - STFT#2 geometry + normalization rule: H1. Q6 fix: calibrated_two_sided_window_sum
    (reference-tone divisor); legacy sum(w)/2 retired (caused peak ~0.029 < 0.05).
  - Lossless storage: exact float32 .npy arrays and inspection .png
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from h1_image import assert_fresh, synth_image

def generate_and_save_image(nfm_item, cfg, config_path, output_dir, save_files=True):
    """
    Generate 128x128 continuous grayscale image from NFM signal.
    Science (synth_image) is unchanged; save_files=False skips legacy
    .npy/.png disk writes so HDF5 stays canonical (storage-only switch).
    """
    assert_fresh(config_path)

    s_bb = nfm_item["s_bb"]
    L_frames = nfm_item["L_frames"]
    vec = nfm_item["vector"]
    cid = vec["contour_id"]

    img, start_col = synth_image(s_bb, L_frames, cfg)

    npy_path = None
    png_path = None
    if save_files:
        npy_dir = os.path.join(output_dir, "npy")
        png_dir = os.path.join(output_dir, "png")
        os.makedirs(npy_dir, exist_ok=True)
        os.makedirs(png_dir, exist_ok=True)

        npy_path = os.path.join(npy_dir, f"{cid}.npy")
        png_path = os.path.join(png_dir, f"{cid}.png")

        # Save exact float32 array
        np.save(npy_path, img)

        # Save PNG grayscale for visual inspection (strictly linear mapping [0, 1] -> [0, 255])
        img_uint8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        plt.imsave(png_path, img_uint8, cmap="gray", vmin=0, vmax=255)
    
    return {
        "contour_id": cid,
        "filename": vec["filename"],
        "channel": vec["channel"],
        "image_shape": list(img.shape),
        "dtype": str(img.dtype),
        "min_val": float(np.min(img)),
        "max_val": float(np.max(img)),
        "mean_val": float(np.mean(img)),
        "start_col": int(start_col),
        "length_frames": int(L_frames),
        "center_freq_hz": float(nfm_item["f_center_hz"]),
        "npy_path": npy_path,
        "png_path": png_path,
        "image_data": img
    }
