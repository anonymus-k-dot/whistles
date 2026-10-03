"""
stage6_vectors.py -- Stage 5: Time-Frequency Vectors Extraction.

Authoritative Reference:
  PREPROCESSING_PIPELINE.md (Section 1-3, D11, D20)
  config_paper_test.yaml (block: vector)

Lineage Categories:
  - Whistle time-frequency vectors: [PAPER-SPECIFIED] Sec. II.C
      "whistle time-frequency vectors were converted into individual whistle images"
  - Center frequency definition (Q4): midrange (fmin+fmax)/2 = H1 (paper: removed/centered only).
  - Dual representation: [IMPLEMENTATION REQUIRED]
      Native (frame_k, bin) and physical (time_s, freq_hz).
"""

import os
import sys
import numpy as np

def extract_time_frequency_vector(contour, cfg):
    """
    Extract dual-representation Time-Frequency vector from accepted contour.
    """
    frame_indices = np.array(contour["frame_indices"], dtype=np.int32)
    bin_coords = np.array(contour["bin_coordinates"], dtype=np.float32)
    times_s = np.array(contour["times_s"], dtype=np.float64)
    freqs_hz = np.array(contour["frequencies_hz"], dtype=np.float64)
    
    L = len(frame_indices)
    f_min = float(np.min(freqs_hz))
    f_max = float(np.max(freqs_hz))
    f_start = float(freqs_hz[0])
    f_end = float(freqs_hz[-1])
    bandwidth = f_max - f_min
    
    # Center frequency per D11: Midrange (fmin + fmax) / 2
    center_def = cfg.get("nfm", {}).get("center_freq_definition", cfg.get("vector", {}).get("center_definition", "midrange"))
    if center_def == "midrange":
        f_center = (f_min + f_max) / 2.0
    elif center_def == "mean":
        f_center = float(np.mean(freqs_hz))
    else:
        raise ValueError(f"Unknown center_definition: {center_def}")
        
    native_rep = np.column_stack([frame_indices, bin_coords]).tolist()
    physical_rep = np.column_stack([times_s, freqs_hz]).tolist()
    
    return {
        "contour_id": contour["contour_id"],
        "filename": contour["filename"],
        "channel": contour["channel"],
        "length_frames": int(L),
        "duration_s": round(float(times_s[-1] - times_s[0]), 4),
        "start_time_s": round(float(times_s[0]), 4),
        "end_time_s": round(float(times_s[-1]), 4),
        "f_min_hz": round(f_min, 2),
        "f_max_hz": round(f_max, 2),
        "f_start_hz": round(f_start, 2),
        "f_end_hz": round(f_end, 2),
        "f_center_hz": round(f_center, 2),
        "bandwidth_hz": round(bandwidth, 2),
        "mean_contrast_db": contour.get("mean_contrast_db", 0.0),
        "native_representation": native_rep,
        "physical_representation": physical_rep
    }
