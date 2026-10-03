"""
h1_image.py -- Hypothesis H1 implementation, derivation, and image synthesis.

Status (Q1-Q6 FINAL DECISIONS 2026-10-03):
  H1 is an [UNRESOLVED IMPLEMENTATION HYPOTHESIS], NOT an established paper fact.
  PAPER FACTS: 192 kHz target, 2 kHz high-pass intent, 2048 Hamming 50%% overlap,
    128x128, continuous [0,1], 0.05 decode threshold.
  H1: resample_poly, Butterworth-4/1791.4Hz/sosfiltfilt, detector 2-20kHz/4dB/gap2/
    jump500/parabolic, min-len 10, L>128 reject, fc midrange, 12 kHz NFM, linear
    interp, cumsum phase, complex exp(jφ), STFT#2 (Hamming128/NFFT128/hop64/
    two-sided/fftshift/centered/zero-pad), image normalization rule, QC tolerances.
  Q6 FIX: legacy "constant_window_sum" (sum(w)/2) retired — it double-normalizes
    under scipy.signal.stft scaling (peak ~0.029 < 0.05). Replacement
    "calibrated_two_sided_window_sum" measures the divisor from a reference
    unit-amplitude DC tone through the IDENTICAL stft call, so a bin-centered
    tone maps to exactly 1.0. Data-independent. Decode threshold 0.05 unchanged.
"""

import os
import sys
import math
import hashlib
import json
import yaml
import numpy as np
import scipy.signal as signal

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_yaml(data, path):
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

def compute_fingerprint(cfg):
    """Compute SHA-256 hash of all configuration inputs that influence H1 derivation."""
    inputs = {
        "nfm_baseband_fs": cfg.get("nfm", {}).get("baseband_fs_hz", cfg.get("h1", {}).get("baseband_fs", 12000.0)),
        "image_size": [cfg.get("image", {}).get("height_rows", 128), cfg.get("image", {}).get("width_cols", 128)],
        "stft1_window": cfg.get("stft1", {}).get("window", "hamming"),
        "stft1_nfft": cfg.get("stft1", {}).get("nfft", 2048),
        "stft1_hop": cfg.get("stft1", {}).get("hop", 1024),
        "resample_fs_out": cfg.get("resample", {}).get("fs_out", 192000),
        "stft2_nfft": cfg.get("image", {}).get("stft2_nfft", cfg.get("h1", {}).get("nfft", 128)),
        "stft2_hop": cfg.get("image", {}).get("stft2_hop", cfg.get("h1", {}).get("hop", 64)),
        "stft2_window": cfg.get("image", {}).get("stft2_window", cfg.get("h1", {}).get("window", "hamming")),
        "normalization": cfg.get("image", {}).get("normalization", "constant_window_sum")
    }
    dumped = json.dumps(inputs, sort_keys=True, default=str)
    return hashlib.sha256(dumped.encode("utf-8")).hexdigest()

def get_window(name, length):
    if name.lower() == "hamming":
        return np.hamming(length)
    elif name.lower() == "hann":
        return np.hanning(length)
    elif name.lower() == "rect" or name.lower() == "rectangular":
        return np.ones(length)
    else:
        raise ValueError(f"Unsupported window: {name}")

def synth_nfm_signal(freq_contour_hz, center_freq_hz, fs_baseband=12000, hop_samples=64, amplitude=1.0):
    """
    Synthesize constant-amplitude NFM complex baseband signal from a contour.
    
    Parameters:
      freq_contour_hz: 1D array of frequency values in Hz (one point per STFT#1 frame)
      center_freq_hz: Midrange center frequency (fmin + fmax) / 2
      fs_baseband: Baseband sample rate (12 kHz)
      hop_samples: STFT#2 hop in samples (64)
      amplitude: Constant envelope (1.0)
      
    Returns:
      s_bb: Complex baseband time signal s(t) = A * exp(j * phi(t))
    """
    L = len(freq_contour_hz)
    f_bb = np.asarray(freq_contour_hz, dtype=np.float64) - float(center_freq_hz)
    
    nperseg = 128
    n_samples = (L - 1) * hop_samples + nperseg
    
    frame_centers = np.arange(L, dtype=np.float64) * hop_samples + (nperseg // 2)
    sample_indices = np.arange(n_samples, dtype=np.float64)
    
    if L > 1:
        f_inst = np.interp(sample_indices, frame_centers, f_bb)
    else:
        f_inst = np.full(n_samples, f_bb[0], dtype=np.float64)
        
    dt = 1.0 / fs_baseband
    phase = 2.0 * np.pi * np.cumsum(f_inst) * dt
    
    s_bb = amplitude * np.exp(1j * phase)
    return s_bb

def synth_image(s_bb, L_frames, cfg):
    """
    Construct 128x128 continuous grayscale image from NFM baseband signal.
    
    Steps:
      1. Zero-pad signal in time domain to 128 frames (start_col = (128 - L) // 2)
      2. STFT #2: window=128 Hamming, hop=64, nfft=128, two-sided fftshifted
      3. Magnitude normalized by sum(window) * amplitude
      4. Output strictly in [0.0, 1.0], shape (128, 128), float32
    """
    nfft = cfg.get("image", {}).get("stft2_nfft", cfg.get("h1", {}).get("nfft", 128))
    nperseg = cfg.get("image", {}).get("stft2_nfft", cfg.get("h1", {}).get("nperseg", 128))
    hop = cfg.get("image", {}).get("stft2_hop", cfg.get("h1", {}).get("hop", 64))
    window_name = cfg.get("image", {}).get("stft2_window", cfg.get("h1", {}).get("window", "hamming"))
    win = get_window(window_name, nperseg)
    amp = cfg.get("nfm", {}).get("amplitude", 1.0)
    
    total_frames = 128
    start_col = (total_frames - L_frames) // 2
    
    total_samples = (total_frames - 1) * hop + nperseg
    full_signal = np.zeros(total_samples, dtype=np.complex128)
    
    start_sample = start_col * hop
    end_sample = start_sample + len(s_bb)
    
    if end_sample > total_samples:
        end_sample = total_samples
        s_bb = s_bb[:end_sample - start_sample]
        
    full_signal[start_sample:end_sample] = s_bb
    
    # 2. STFT #2
    f, t, Zxx = signal.stft(
        full_signal,
        fs=cfg.get("nfm", {}).get("baseband_fs_hz", 12000),
        window=win,
        nperseg=nperseg,
        noverlap=nperseg - hop,
        nfft=nfft,
        return_onesided=False,
        boundary=None,
        padded=False
    )
    
    # Frequency axis is [-Fs/2, Fs/2) via fftshift
    Zxx_shifted = np.fft.fftshift(Zxx, axes=0)
    mag = np.abs(Zxx_shifted)
    
    # 3. Normalisation (Q6 FIX): calibrated two-sided reference-tone divisor.
    # Legacy "constant_window_sum" divided by (sum(w)/2 * amp), which assumed an
    # unnormalized FFT (peak = sum(w)). scipy.signal.stft already scales the
    # spectrum, so that divisor double-normalized -> peak ~0.029 < 0.05 decode.
    # New rule "calibrated_two_sided_window_sum" (H1): pass a unit-amplitude DC
    # reference tone through the IDENTICAL stft call and divide by its measured
    # peak. Guarantees a bin-centered tone -> 1.0 on any scipy version.
    # Data-independent: reference is synthetic ones(), never dataset statistics.
    norm_mode = cfg.get("image", {}).get("normalization", "calibrated_two_sided_window_sum")
    if norm_mode == "constant_window_sum":
        # Legacy path retained for reproducibility only; NOT used by default config.
        norm_factor = (np.sum(win) / 2.0) * amp
        img = mag / norm_factor
    elif norm_mode in ("calibrated_two_sided_window_sum", "constant_window_sum_calibrated",
                        "reference_tone_calibrated"):
        ref_len = max(int(nperseg * 8), 1024)
        ref_tone = np.ones(ref_len, dtype=np.complex128)  # DC, amplitude 1.0
        _, _, Zref = signal.stft(
            ref_tone,
            fs=cfg.get("nfm", {}).get("baseband_fs_hz", 12000),
            window=win,
            nperseg=nperseg,
            noverlap=nperseg - hop,
            nfft=nfft,
            return_onesided=False,
            boundary=None,
            padded=False
        )
        Zref_shifted = np.fft.fftshift(Zref, axes=0)
        ref_peak = float(np.max(np.abs(Zref_shifted)))
        if not np.isfinite(ref_peak) or ref_peak <= 0:
            raise ValueError(f"Reference-tone calibration failed (ref_peak={ref_peak})")
        norm_factor = ref_peak * amp
        img = mag / norm_factor
    else:
        raise ValueError(f"Unknown image normalization: {norm_mode}")
    
    img = np.clip(img, 0.0, 1.0).astype(np.float32)
    
    # Ensure shape is exactly 128x128
    if img.shape[1] > 128:
        img = img[:, :128]
    elif img.shape[1] < 128:
        pad_cols = 128 - img.shape[1]
        img = np.pad(img, ((0, 0), (0, pad_cols)), mode='constant')
        
    return img, start_col

def decode_image(img, threshold=0.05, bin_hz=93.75, center_freq_hz=0.0, center_row=64):
    """
    Decode whistle contour from 128x128 image using thresholding and sub-bin centroid.
    Used for QC and contour recovery only.
    """
    active_cols = []
    recovered_freqs = []
    recovered_rows = []
    
    rows, cols = img.shape
    row_indices = np.arange(rows, dtype=np.float64)
    
    for c in range(cols):
        col_vals = img[:, c]
        mask = col_vals > threshold
        if np.any(mask):
            sub_vals = col_vals[mask]
            sub_rows = row_indices[mask]
            centroid_row = np.sum(sub_rows * sub_vals) / np.sum(sub_vals)
            
            f_bb = (centroid_row - center_row) * bin_hz
            f_hz = f_bb + center_freq_hz
            
            active_cols.append(c)
            recovered_rows.append(centroid_row)
            recovered_freqs.append(f_hz)
            
    return np.array(active_cols), np.array(recovered_freqs), np.array(recovered_rows)

def run_synthetic_simulation(cfg):
    """
    Run synthetic contours across diverse shapes to verify H1 reconstruction.
    """
    fs_baseband = cfg.get("nfm", {}).get("baseband_fs_hz", cfg.get("h1", {}).get("baseband_fs", 12000.0))
    hop_bb = cfg.get("image", {}).get("stft2_hop", cfg.get("h1", {}).get("hop", 64))
    bin_hz = cfg.get("stft1", {}).get("bin_hz", 93.75)
    nfft = cfg.get("image", {}).get("stft2_nfft", cfg.get("h1", {}).get("nfft", 128))
    center_row = cfg.get("image", {}).get("center_row", nfft // 2)
    
    synthetic_cases = [
        {"name": "shallow_horizontal", "f_start": 8000, "f_end": 8000, "length": 40, "type": "linear"},
        {"name": "slow_rising", "f_start": 6000, "f_end": 8000, "length": 50, "type": "linear"},
        {"name": "slow_falling", "f_start": 9000, "f_end": 6500, "length": 45, "type": "linear"},
        {"name": "steep_rising", "f_start": 4000, "f_end": 10000, "length": 25, "type": "linear"},
        {"name": "concave_curved", "f_start": 5000, "f_end": 8500, "length": 55, "type": "parabolic"},
        {"name": "convex_sinusoid", "f_start": 7000, "f_end": 7000, "length": 60, "type": "sinusoid"},
        {"name": "short_whistle", "f_start": 7500, "f_end": 8500, "length": 15, "type": "linear"},
        {"name": "long_whistle", "f_start": 5500, "f_end": 9500, "length": 120, "type": "linear"}
    ]
    
    results = []
    max_centroid_err = 0.0
    max_len_extension = 0
    
    for case in synthetic_cases:
        L = case["length"]
        t = np.linspace(0, 1, L)
        if case["type"] == "linear":
            freqs = case["f_start"] + (case["f_end"] - case["f_start"]) * t
        elif case["type"] == "parabolic":
            freqs = case["f_start"] + (case["f_end"] - case["f_start"]) * (t ** 2)
        elif case["type"] == "sinusoid":
            freqs = case["f_start"] + 1500.0 * np.sin(np.pi * t)
            
        fmin = np.min(freqs)
        fmax = np.max(freqs)
        f_mid = (fmin + fmax) / 2.0
        
        gt_bb_bins = (freqs - f_mid) / bin_hz
        gt_rows = center_row + gt_bb_bins
        
        s_bb = synth_nfm_signal(freqs, f_mid, fs_baseband=fs_baseband, hop_samples=hop_bb)
        img, start_col = synth_image(s_bb, L, cfg)
        
        dec_cols, dec_freqs, dec_rows = decode_image(img, threshold=0.05, bin_hz=bin_hz, center_freq_hz=f_mid, center_row=center_row)
        
        dec_len = len(dec_cols)
        len_ext = dec_len - L
        if len_ext > max_len_extension:
            max_len_extension = len_ext
            
        c_errs = []
        for k in range(L):
            col_target = start_col + k
            idx = np.where(dec_cols == col_target)[0]
            if len(idx) > 0:
                err = abs(dec_rows[idx[0]] - gt_rows[k])
                c_errs.append(err)
            else:
                c_errs.append(1.0)
                
        case_max_c_err = max(c_errs) if c_errs else 0.0
        if case_max_c_err > max_centroid_err:
            max_centroid_err = case_max_c_err
            
        results.append({
            "name": case["name"],
            "length": L,
            "max_centroid_error_bins": round(float(case_max_c_err), 4),
            "mean_centroid_error_bins": round(float(np.mean(c_errs)), 4),
            "decoded_length": int(dec_len),
            "length_extension_frames": int(len_ext),
            "peak_intensity": round(float(np.max(img)), 4)
        })
        
    return results, max_centroid_err, max_len_extension

def derive_h1(config_path, output_derived_path=None):
    """Derive H1 parameters and write to derived_h1.yaml."""
    cfg = load_yaml(config_path)
    if output_derived_path is None:
        derived_filename = cfg.get("paths", {}).get("derived_h1_path", "derived_h1.yaml")
        output_derived_path = os.path.join(os.path.dirname(config_path), derived_filename)
        output_derived_path = os.path.abspath(output_derived_path)
        
    sim_results, sim_max_c_err, sim_max_len_ext = run_synthetic_simulation(cfg)
    
    nfft = cfg.get("image", {}).get("stft2_nfft", cfg.get("h1", {}).get("nfft", 128))
    margin = cfg.get("support", {}).get("baseband_margin_bins", 2)
    center_row = cfg.get("image", {}).get("center_row", nfft // 2)
    max_abs_bins = center_row - margin
    
    fs_bb = cfg.get("nfm", {}).get("baseband_fs_hz", cfg.get("h1", {}).get("baseband_fs", 12000.0))
    hop_bb = cfg.get("image", {}).get("stft2_hop", cfg.get("h1", {}).get("hop", 64))
    col_per_frame = (hop_bb / fs_bb) / (cfg["stft1"]["hop"] / cfg["resample"]["fs_out"])
    if not math.isclose(col_per_frame, 1.0, rel_tol=1e-5):
        raise ValueError(f"H1 framing mismatch: columns_per_stft1_frame = {col_per_frame} != 1.0")
        
    fingerprint = compute_fingerprint(cfg)
    
    qc_centroid_tol = cfg.get("validation", {}).get("max_centroid_error_bins", 1.0)
    qc_length_tol = cfg.get("validation", {}).get("max_length_error_frames", 3)
    
    derived_data = {
        "status": "H1-derived parameters. Auto-generated by h1_image.py. Do not edit by hand.",
        "fingerprint": fingerprint,
        "derivation_timestamp": str(np.datetime64('now')),
        "center_row_0based": int(center_row),
        "max_abs_baseband_bins": int(max_abs_bins),
        "columns_per_stft1_frame": float(col_per_frame),
        "simulated_max_centroid_error_bins": round(float(sim_max_c_err), 4),
        "simulated_max_length_extension_frames": int(sim_max_len_ext),
        "qc_centroid_tolerance_bins": float(qc_centroid_tol),
        "qc_length_tolerance_frames": int(qc_length_tol),
        "simulation_summary": sim_results
    }
    
    save_yaml(derived_data, output_derived_path)
    return derived_data

def assert_fresh(config_path, derived_path=None):
    """Assert derived_h1.yaml exists and matches config fingerprint."""
    cfg = load_yaml(config_path)
    if derived_path is None:
        derived_filename = cfg.get("paths", {}).get("derived_h1_path", "derived_h1.yaml")
        derived_path = os.path.join(os.path.dirname(config_path), derived_filename)
        
    if not os.path.exists(derived_path):
        # Auto-derive if missing
        return derive_h1(config_path, derived_path)
        
    derived_data = load_yaml(derived_path)
    current_fp = compute_fingerprint(cfg)
    stored_fp = derived_data.get("fingerprint")
    
    if current_fp != stored_fp:
        # Re-derive automatically
        return derive_h1(config_path, derived_path)
        
    return derived_data
