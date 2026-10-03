"""
stage9_qc.py -- Internal Implementation Validation & Quality Control.

Authoritative Reference:
  PREPROCESSING_PIPELINE.md (Section 1-4, D16, D17)
  config_paper_test.yaml (block: validation)

Lineage Categories:
  - 0.05 contour decode threshold: [PAPER-SPECIFIED] Fig. 2(b) (Q6, DO NOT CHANGE).
  - QC tolerances 1 bin / 3 frames: H1 / ENGINEERING QC ONLY (Q6), not paper method.
      CRITICAL: Round-trip QC is NOT part of the research paper's preprocessing!
      It is an internal engineering diagnostic to verify image synthesis fidelity.
"""

import os
import sys
import numpy as np
from h1_image import assert_fresh, decode_image

def perform_roundtrip_qc(img_item, nfm_item, cfg, config_path):
    """
    Execute round-trip verification on a single generated image:
      1. Integrity checks (finite, range [0, 1], shape (128, 128))
      2. Decode contour using 0.05 threshold (Fig. 2(b))
      3. Compare recovered vs original contour
    """
    derived_h1 = assert_fresh(config_path)
    
    img = img_item["image_data"]
    vec = nfm_item["vector"]
    cid = vec["contour_id"]
    L_orig = nfm_item["L_frames"]
    f_center = nfm_item["f_center_hz"]
    start_col = img_item["start_col"]
    
    bin_hz = cfg.get("stft1", {}).get("bin_hz", 93.75)
    decode_thresh = cfg.get("validation", {}).get("contour_decode_threshold", cfg.get("validation", {}).get("decode_threshold", 0.05))
    center_row = derived_h1.get("center_row_0based", 64)
    
    max_c_tol = derived_h1.get("qc_centroid_tolerance_bins", 1)
    max_l_tol = derived_h1.get("qc_length_tolerance_frames", 3)
    
    integrity_failed = False
    integrity_errors = []
    
    if img.shape != (128, 128):
        integrity_failed = True
        integrity_errors.append(f"Invalid shape {img.shape} != (128, 128)")
    if not np.all(np.isfinite(img)):
        integrity_failed = True
        integrity_errors.append("Non-finite values (NaN/Inf) detected")
    if np.min(img) < 0.0 or np.max(img) > 1.000001:
        integrity_failed = True
        integrity_errors.append(f"Values outside range [0, 1]: min={np.min(img)}, max={np.max(img)}")
        
    if integrity_failed:
        return {
            "contour_id": cid,
            "status": "integrity_failure",
            "errors": integrity_errors,
            "max_centroid_error_bins": None,
            "length_error_frames": None,
            "recovered_length": None,
            "original_length": L_orig
        }
        
    dec_cols, dec_freqs, dec_rows = decode_image(
        img,
        threshold=decode_thresh,
        bin_hz=bin_hz,
        center_freq_hz=f_center,
        center_row=center_row
    )
    
    L_recovered = len(dec_cols)
    length_error = abs(L_recovered - L_orig)
    
    phys = np.array(vec["physical_representation"])
    orig_freqs = phys[:, 1]
    gt_rows = center_row + (orig_freqs - f_center) / bin_hz
    
    centroid_errors = []
    for k in range(L_orig):
        col_target = start_col + k
        idx = np.where(dec_cols == col_target)[0]
        if len(idx) > 0:
            c_err = abs(dec_rows[idx[0]] - gt_rows[k])
            centroid_errors.append(c_err)
        else:
            centroid_errors.append(2.0)
            
    max_c_err = float(np.max(centroid_errors)) if centroid_errors else 999.0
    mean_c_err = float(np.mean(centroid_errors)) if centroid_errors else 999.0
    
    fidelity_failed = (max_c_err > max_c_tol) or (length_error > max_l_tol)
    rejection_reasons = []
    if max_c_err > max_c_tol:
        rejection_reasons.append(f"Centroid error {max_c_err:.3f} bins > tolerance {max_c_tol} bins")
    if length_error > max_l_tol:
        rejection_reasons.append(f"Length error {length_error} frames > tolerance {max_l_tol} frames")
        
    status = "fidelity_failure" if fidelity_failed else "pass"
    
    return {
        "contour_id": cid,
        "filename": vec["filename"],
        "channel": vec["channel"],
        "status": status,
        "original_length": int(L_orig),
        "recovered_length": int(L_recovered),
        "length_error_frames": int(length_error),
        "max_centroid_error_bins": round(max_c_err, 4),
        "mean_centroid_error_bins": round(mean_c_err, 4),
        "peak_intensity": round(float(np.max(img)), 4),
        "rejection_reasons": rejection_reasons,
        "decoded_cols": dec_cols.tolist(),
        "decoded_freqs_hz": dec_freqs.tolist(),
        "decoded_rows": dec_rows.tolist()
    }
