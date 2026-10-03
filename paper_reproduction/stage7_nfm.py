"""
stage7_nfm.py -- Stage 6: Whistle Support Screening (>128 Frames) & Constant-Amplitude NFM Synthesis.

Authoritative Reference:
  PREPROCESSING_PIPELINE.md (Section 1-3, D9, D10, D11, D12)
  config_paper_test.yaml (blocks: support, vector, h1, nfm)

Lineage Categories:
  - Constant-amplitude NFM baseband signal: [PAPER-SPECIFIED] (Q5 method fact).
    H1 realization: 12 kHz, linear interp, float64 cumsum phase, complex exp(jφ), amp 1.0.
  - Whistles > 128 frames (Q4): reject + log = H1 (paper fixes width only).
      128 column width is paper-specified; policy for overlong whistles is unstated.
      Dropping and logging is the safest neutral policy to preserve frequency derivative integrity.
"""

import os
import sys
import numpy as np
from h1_image import assert_fresh, synth_nfm_signal

def screen_and_synthesize_nfm(tf_vectors, cfg, config_path):
    """
    Screen contours for 128-frame support and synthesize constant-amplitude NFM signals.
    """
    derived_h1 = assert_fresh(config_path)
    
    max_len = cfg.get("long_contours", cfg.get("support", {})).get("max_length_frames", 128)
    max_bins = derived_h1.get("max_abs_baseband_bins", 62)
    bin_hz = cfg.get("stft1", {}).get("bin_hz", 93.75)
    max_abs_hz = max_bins * bin_hz
    
    fs_bb = cfg.get("nfm", {}).get("baseband_fs_hz", cfg.get("h1", {}).get("baseband_fs", 12000.0))
    hop_bb = cfg.get("image", {}).get("stft2_hop", cfg.get("h1", {}).get("hop", 64))
    amp = cfg.get("nfm", {}).get("amplitude", 1.0)
    
    supported_nfm = []
    overlong_records = []
    out_of_margin_records = []
    
    total_in = len(tf_vectors)
    
    for vec in tf_vectors:
        L = vec["length_frames"]
        phys = np.array(vec["physical_representation"])
        freqs_hz = phys[:, 1]
        f_center = vec["f_center_hz"]
        
        # Check condition 1: Length <= 128 frames
        if L > max_len:
            ov = dict(vec)
            ov["reason"] = "length_exceeds_128_frames"
            ov["max_length_limit"] = max_len
            ov["excess_frames"] = L - max_len
            overlong_records.append(ov)
            continue
            
        # Check condition 2: Baseband excursion <= max_abs_baseband_bins (62 bins = 5812.5 Hz)
        max_dev_hz = float(np.max(np.abs(freqs_hz - f_center)))
        if max_dev_hz > max_abs_hz:
            oom = dict(vec)
            oom["reason"] = "exceeds_baseband_margin"
            oom["max_deviation_hz"] = round(max_dev_hz, 2)
            oom["max_allowed_hz"] = round(max_abs_hz, 2)
            out_of_margin_records.append(oom)
            continue
            
        s_bb = synth_nfm_signal(
            freq_contour_hz=freqs_hz,
            center_freq_hz=f_center,
            fs_baseband=fs_bb,
            hop_samples=hop_bb,
            amplitude=amp
        )
        
        supported_nfm.append({
            "vector": vec,
            "s_bb": s_bb,
            "L_frames": L,
            "f_center_hz": f_center,
            "baseband_fs": fs_bb
        })
        
    dropped_overlong = len(overlong_records)
    dropped_margin = len(out_of_margin_records)
    supported_count = len(supported_nfm)
    
    overlong_pct = (dropped_overlong / total_in * 100.0) if total_in > 0 else 0.0
    review_trigger_pct = cfg.get("long_contours", cfg.get("support", {})).get("review_trigger_dropped_fraction", 0.05) * 100.0
    trigger_exceeded = overlong_pct > review_trigger_pct
    
    screening_summary = {
        "total_input_contours": total_in,
        "supported_contours": supported_count,
        "overlong_contours_dropped": dropped_overlong,
        "overlong_percentage": round(overlong_pct, 2),
        "review_trigger_percentage": review_trigger_pct,
        "review_trigger_exceeded": trigger_exceeded,
        "max_length_frames_limit": max_len,
        "out_of_margin_dropped": dropped_margin
    }
    
    return supported_nfm, overlong_records, out_of_margin_records, screening_summary
