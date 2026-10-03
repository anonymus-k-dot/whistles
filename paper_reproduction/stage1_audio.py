"""
stage1_audio.py -- Stage 1: Raw Audio Validation, Per-Channel QC, Resampling, and High-Pass Filtering.

Authoritative Reference:
  PREPROCESSING_PIPELINE.md (Sections 1-3, D1, D2, D3, D20)
  config_paper_test.yaml (blocks: input, channels, time, resample, highpass)

Lineage Categories:
  - 500 kHz sample rate: [PAPER-SPECIFIED] Sec. II.C
  - 6 channels (hydrophones): [PAPER-SPECIFIED] Sec. II.C
  - ~60 s duration: [PAPER-SPECIFIED] Sec. II.C
  - Channel handling: PAPER FACT — the source recordings are validated as
    6-channel recordings, but only channel 0 (the first hydrophone/channel)
    is used by the reproduction pipeline. Channels 1-5 are never resampled,
    filtered, or processed downstream.
  - Resampling 500 kHz -> 192 kHz: [PAPER-SPECIFIED] target rate (Q2).
    H1: Polyphase Kaiser method (ratio 48/125) — paper silent on algorithm.
  - High-pass filter edge 2 kHz: [PAPER-SPECIFIED] passband above 2 kHz (Q2 intent only).
    H1: 4th-order Butterworth, zero-phase (sosfiltfilt), design ~1791.4 Hz.
    *** CRITICAL SCIENTIFIC DISTINCTION ***:
    The design cutoff frequency ~1791.4 Hz is NOT a parameter from the paper!
    It is our implementation realization to achieve -3.01 dB at the paper's 2000 Hz edge
    under zero-phase forward-backward filtering:
        f_design = f_edge * (sqrt(2) - 1)**(1 / (2 * order)) ≈ 1791.439 Hz.
"""

import os
import sys
import glob
import math
import csv
import json
import yaml
import soundfile as sf
import numpy as np
import scipy.signal as signal

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def design_cutoff_hz(passband_edge_hz=2000.0, order=4, corner_spec="minus3db_at_passband_edge"):
    """
    Compute Butterworth filter design cutoff frequency.
    CRITICAL DISTINCTION:
      f_design (~1791.4 Hz for n=4) is an [IMPLEMENTATION REQUIRED] realization,
      NOT a parameter explicitly given by the paper. The paper specifies ONLY
      'passband above 2 kHz' [PAPER-SPECIFIED].
    """
    if corner_spec == "minus3db_at_passband_edge":
        f_design = passband_edge_hz * ((math.sqrt(2.0) - 1.0) ** (1.0 / (2.0 * order)))
        return float(f_design)
    elif corner_spec == "design_cutoff_equals_edge":
        return float(passband_edge_hz)
    else:
        raise ValueError(f"Unknown corner_spec: {corner_spec}")

def validate_raw_audio(file_path, cfg):
    """
    Inspect raw FLAC file against invariants without modifying it.
    Returns validation dict, per-channel statistics list, and raw int16 data.
    """
    info = sf.info(file_path)
    warn_dur_range = cfg["input"].get("duration_warn_range_s", [58.0, 62.0])
    warn_min_dur, warn_max_dur = warn_dur_range
    clip_thresh = 0.001
    dead_rms_thresh = 1.0
    expected_fs = cfg["input"].get("expected_fs", 500000)
    expected_channels = cfg["input"].get("expected_channels", 6)
    
    validation = {
        "filename": os.path.basename(file_path),
        "path": file_path,
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "frames": info.frames,
        "duration_s": round(info.duration, 4),
        "format": info.format,
        "subtype": info.subtype,
        "fs_valid": (info.samplerate == expected_fs),
        "channels_valid": (info.channels == expected_channels),
        "duration_in_warn_range": (warn_min_dur <= info.duration <= warn_max_dur),
        "is_finite": True,
        "has_clipping_warning": False,
        "has_dead_channel_warning": False,
        "error_messages": []
    }
    
    if not validation["fs_valid"]:
        validation["error_messages"].append(f"Sample rate mismatch: {info.samplerate} != {expected_fs}")
    if not validation["channels_valid"]:
        validation["error_messages"].append(f"Channel count mismatch: {info.channels} != {expected_channels}")
    if not validation["duration_in_warn_range"]:
        validation["error_messages"].append(f"Duration {info.duration:.2f}s outside nominal range [{warn_min_dur}, {warn_max_dur}]s")
        
    data_int16, sr = sf.read(file_path, dtype="int16", always_2d=True)
    
    if data_int16.shape[1] != info.channels:
        validation["error_messages"].append(f"Decoded channel count mismatch: {data_int16.shape[1]} != {info.channels}")
        
    ch_stats = []
    for ch_idx in range(data_int16.shape[1]):
        ch_raw = data_int16[:, ch_idx]
        
        if not np.all(np.isfinite(ch_raw)):
            validation["is_finite"] = False
            validation["error_messages"].append(f"Channel {ch_idx} contains non-finite values (NaN/Inf)")
            
        n_samples = len(ch_raw)
        rms = float(np.sqrt(np.mean(ch_raw.astype(np.float64) ** 2)))
        peak = int(np.max(np.abs(ch_raw)))
        clipped_count = int(np.sum(np.abs(ch_raw) >= 32767))
        clip_frac = float(clipped_count / n_samples)
        is_dead = (rms < dead_rms_thresh)
        is_clipped = (clip_frac > clip_thresh)
        
        if is_dead:
            validation["has_dead_channel_warning"] = True
            validation["error_messages"].append(f"Channel {ch_idx} dead (RMS {rms:.2f} < {dead_rms_thresh} counts)")
        if is_clipped:
            validation["has_clipping_warning"] = True
            validation["error_messages"].append(f"Channel {ch_idx} clipped ({clipped_count} samples, {clip_frac*100:.3f}%)")
            
        ch_stats.append({
            "filename": os.path.basename(file_path),
            "channel_idx": ch_idx,
            "n_samples": n_samples,
            "duration_s": round(n_samples / sr, 4),
            "rms_int16": round(rms, 2),
            "peak_int16": peak,
            "clipped_samples": clipped_count,
            "clip_fraction": round(clip_frac, 6),
            "mean_int16": round(float(np.mean(ch_raw)), 2),
            "std_int16": round(float(np.std(ch_raw)), 2),
            "min_int16": int(np.min(ch_raw)),
            "max_int16": int(np.max(ch_raw)),
            "is_dead": is_dead,
            "is_clipped": is_clipped
        })
        
    validation["passed"] = (validation["fs_valid"] and validation["channels_valid"] and validation["is_finite"])
    return validation, ch_stats, data_int16

def process_file_stage1(file_path, cfg):
    """
    Execute Stage 1 pipeline on one file (PERFORMANCE-OPTIMIZED, channel-0 only):
      1. Validation of 6-channel source & per-channel stats (all channels, read-only)
      2. Select channel 0 immediately after validation: audio[:, 0]
      3. Polyphase resampling 500 kHz -> 192 kHz (ratio 48/125) on channel 0 ONLY
      4. Zero-phase Butterworth high-pass filtering on channel 0 ONLY
      Channels 1-5 are NEVER resampled, filtered, or otherwise processed.

    Returns:
      audio_192k_hp_ch0: np.ndarray shape (n_samples_192k,), dtype float32 (1D, channel 0)
      meta: dict with all validation, stats, filter parameters + selected_channel=0
    """
    validation, ch_stats, data_int16 = validate_raw_audio(file_path, cfg)

    if not validation["passed"]:
        print(f"WARNING: File {validation['filename']} violated invariants: {validation['error_messages']}")

    if data_int16.shape[1] < 1:
        raise ValueError(f"Channel-0 selection failed: file has {data_int16.shape[1]} channels")

    # Select channel 0 (first hydrophone) immediately after validation.
    # PAPER FACT: reproduction pipeline uses only channel 0. No averaging,
    # concatenation, beamforming, or cross-channel processing.
    ch0_int16 = data_int16[:, 0]
    # Release the full 6-channel buffer; channels 1-5 leave scope here.
    del data_int16

    data_float64 = ch0_int16.astype(np.float64) / 32768.0
    del ch0_int16
    
    up = cfg["resample"].get("polyphase_up", cfg["resample"].get("up", 48))
    down = cfg["resample"].get("polyphase_down", cfg["resample"].get("down", 125))
    fs_in = cfg["input"]["expected_fs"]
    fs_out = cfg["resample"]["fs_out"]

    # Resample channel 0 ONLY (1D, axis=0). Channels 1-5 are not in scope.
    resampled = signal.resample_poly(data_float64, up, down, axis=0)
    n_resampled_samples = resampled.shape[0]
    
    edge_hz = cfg["highpass"]["passband_edge_hz"] # 2000.0 [PAPER-SPECIFIED]
    order = cfg["highpass"]["order"] # 4 [IMPLEMENTATION REQUIRED]
    corner_spec = cfg["highpass"].get("corner_spec", "minus3db_at_passband_edge")
    f_cutoff = design_cutoff_hz(edge_hz, order=order, corner_spec=corner_spec)
    
    sos = signal.butter(order, f_cutoff, btype="highpass", fs=fs_out, output="sos")
    # Filter channel 0 ONLY. No operation touches channels 1-5 beyond this point
    # (they were released from scope above).
    filtered = signal.sosfiltfilt(sos, resampled, axis=0)

    audio_192k_hp_ch0 = filtered.astype(np.float32)

    meta = {
        "filename": validation["filename"],
        "source_path": file_path,
        "selected_channel": 0,
        "selection_note": "PAPER FACT: source validated as 6-channel; only channel 0 processed downstream.",
        "raw_validation": validation,
        "channel_statistics": ch_stats,
        "dsp_record": {
            "fs_in": fs_in,
            "fs_out": fs_out,
            "resample_ratio": f"{up}/{down}",
            "resample_method": cfg["resample"]["method"],
            "channels_resampled": 1,
            "channels_ignored": [1, 2, 3, 4, 5],
            "raw_samples_per_ch": int(validation["frames"]),
            "resampled_samples_per_ch": int(n_resampled_samples),
            "resampled_duration_s": round(n_resampled_samples / fs_out, 4),
            "highpass_passband_edge_hz": edge_hz,
            "highpass_design_cutoff_hz": round(f_cutoff, 4),
            "highpass_order": order,
            "highpass_type": "butterworth_sosfiltfilt_zero_phase",
            "highpass_note": "Design cutoff 1791.4 Hz is an [IMPLEMENTATION REQUIRED] realization of -3 dB at 2 kHz, NOT a paper parameter."
        }
    }
    
    return audio_192k_hp_ch0, meta
