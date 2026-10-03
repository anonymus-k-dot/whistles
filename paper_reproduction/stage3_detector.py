"""
stage3_detector.py -- Stage 3: Whistle Contour Detection Interface & Stand-In Substitution.

Authoritative Reference:
  PREPROCESSING_PIPELINE.md (Section 1-3, D5, D6)
  config_paper_test.yaml (block: detector)

Lineage Categories:
  - Paper detector: [PAPER-SPECIFIED]
      Paper Section II.C states:
      "In the second stage, a neural network-based whistle extraction method [31]
       was applied to detect individual whistle contours. The extraction process was
       combined with a post-processing algorithm [32], [33] that converts detection
       outputs into individual whistle tracks."
  - Reference [31]: G.-H. Park et al., "Implementation of a neural network for cetacean
      whistle frequency detection based on data synthesis," Proc. KICS Winter Conf., 2024.
  - Reference [32]: G.-H. Park et al., "A design of MCA-CFAR detector for extraction of
      cetacean whistle contours," Proc. KICS Winter Conf., 2024.
  - Reference [33]: G.-H. Park et al., "Detection and tracking-based dolphin whistle
      extraction method," Proc. Korea Inst. Mil. Sci. Technol. Conf., 2024.
  - Status: [ENGINEERING SUBSTITUTION] / H1 DETECTOR (Q3 FINAL DECISION).
      2-20 kHz band, 4 dB, gap 2, 500 Hz jump, parabolic = H1, NOT paper facts.
      References [31]-[33] are 2-page Korean conference extended abstracts.
      No source code, neural network architecture details, trained model weights,
      training data, or hyperparameter tables are published or available in the project.
      The neural detector CANNOT be reconstructed without inventing missing weights/architecture.
  - Test implementation: Classical Ridge Tracker stand-in.
      Classified as [ENGINEERING SUBSTITUTION].
      DO NOT call this the paper's detector.
"""

import os
import sys
import numpy as np

def apply_detector_transform(S_db, transform_name="per_bin_median_subtraction_db"):
    """Apply detector-side background subtraction transform to log-magnitude spectrogram."""
    n_bins, n_frames = S_db.shape
    if transform_name == "none":
        return S_db.copy(), np.zeros(n_bins, dtype=np.float32)
    elif transform_name == "per_bin_median_subtraction_db":
        bin_medians = np.median(S_db, axis=1, keepdims=True)
        S_norm = S_db - bin_medians
        return S_norm.astype(np.float32), bin_medians.squeeze()
    else:
        raise ValueError(f"Unknown input_transform: {transform_name}")

def parabolic_subbin_interpolation(alpha, beta, gamma, bin_idx, bin_hz):
    """Refine peak frequency using 3-point parabolic interpolation on dB values."""
    denom = alpha - 2.0 * beta + gamma
    if abs(denom) < 1e-12:
        delta = 0.0
    else:
        delta = 0.5 * (alpha - gamma) / denom
        delta = np.clip(delta, -0.5, 0.5)
    return float((bin_idx + delta) * bin_hz)

def detect_whistle_contours(
    S_db,
    times_s,
    freqs_hz,
    cfg,
    transform_name=None,
    file_id="",
    channel_idx=0
):
    """
    Extract candidate whistle contours using the stand-in tracker.
    
    Returns:
      raw_contours: List of candidate whistle tracks
      meta: Execution metadata documenting substitution status
    """
    det_cfg = cfg["detector"]
    transform = transform_name or det_cfg.get("input_transform", "per_bin_median_subtraction_db")
    
    # 1. Background subtraction
    S_norm, bg_profile = apply_detector_transform(S_db, transform_name=transform)
    
    # 2. Band filtering [2000, 20000] Hz
    f_min, f_max = det_cfg.get("band_hz", [2000, 20000])
    bin_hz = cfg["stft1"]["bin_hz"]
    
    bin_start = int(np.ceil(f_min / bin_hz))
    bin_end = int(np.floor(f_max / bin_hz))
    bin_start = max(1, bin_start)
    bin_end = min(len(freqs_hz) - 2, bin_end)
    
    min_contrast_db = det_cfg.get("min_contrast_db", 9.0)
    max_gap_frames = det_cfg.get("max_gap_frames", 2)
    max_freq_jump_hz = 500.0 # ~5 bins per frame (~93.75 kHz/s)
    
    n_bins, n_frames = S_norm.shape
    
    # Spectral peak detection
    peaks_per_frame = []
    for k in range(n_frames):
        col = S_norm[:, k]
        interior = col[bin_start:bin_end + 1]
        left = col[bin_start - 1:bin_end]
        right = col[bin_start + 1:bin_end + 2]
        
        is_peak = (interior > left) & (interior > right) & (interior >= min_contrast_db)
        peak_bins = np.where(is_peak)[0] + bin_start
        
        frame_peaks = []
        for b in peak_bins:
            alpha = float(col[b - 1])
            beta = float(col[b])
            gamma = float(col[b + 1])
            sub_f = parabolic_subbin_interpolation(alpha, beta, gamma, b, bin_hz)
            frame_peaks.append({
                "bin": int(b),
                "sub_f_hz": sub_f,
                "contrast_db": beta
            })
        peaks_per_frame.append(frame_peaks)
        
    active_tracks = []
    finished_tracks = []
    track_counter = 0
    
    for k in range(n_frames):
        f_peaks = peaks_per_frame[k]
        t_k = float(times_s[k])
        
        matched_track_indices = set()
        matched_peak_indices = set()
        
        candidate_pairs = []
        for t_idx, track in enumerate(active_tracks):
            for p_idx, peak in enumerate(f_peaks):
                df = abs(peak["sub_f_hz"] - track["last_f"])
                if df <= max_freq_jump_hz:
                    candidate_pairs.append((df, t_idx, p_idx))
                    
        candidate_pairs.sort(key=lambda x: x[0])
        
        for df, t_idx, p_idx in candidate_pairs:
            if t_idx in matched_track_indices or p_idx in matched_peak_indices:
                continue
            matched_track_indices.add(t_idx)
            matched_peak_indices.add(p_idx)
            
            p = f_peaks[p_idx]
            track = active_tracks[t_idx]
            
            gap_len = k - track["last_k"] - 1
            if gap_len > 0:
                f_prev = track["last_f"]
                f_curr = p["sub_f_hz"]
                c_prev = track["contrasts"][-1]
                c_curr = p["contrast_db"]
                for g in range(1, gap_len + 1):
                    gk = track["last_k"] + g
                    frac = g / (gap_len + 1)
                    track["frames"].append(gk)
                    track["times"].append(float(times_s[gk]))
                    track["freqs"].append(f_prev + frac * (f_curr - f_prev))
                    track["bins"].append((f_prev + frac * (f_curr - f_prev)) / bin_hz)
                    track["contrasts"].append(c_prev + frac * (c_curr - c_prev))
                    
            track["frames"].append(k)
            track["times"].append(t_k)
            track["freqs"].append(p["sub_f_hz"])
            track["bins"].append(p["sub_f_hz"] / bin_hz)
            track["contrasts"].append(p["contrast_db"])
            track["last_k"] = k
            track["last_f"] = p["sub_f_hz"]
            track["gap"] = 0
            
        surviving_active = []
        for t_idx, track in enumerate(active_tracks):
            if t_idx not in matched_track_indices:
                track["gap"] += 1
                if track["gap"] <= max_gap_frames:
                    surviving_active.append(track)
                else:
                    finished_tracks.append(track)
            else:
                surviving_active.append(track)
        active_tracks = surviving_active
        
        for p_idx, peak in enumerate(f_peaks):
            if p_idx not in matched_peak_indices:
                track_counter += 1
                new_track = {
                    "track_id": f"{file_id}_ch{channel_idx}_t{track_counter:04d}",
                    "frames": [k],
                    "times": [t_k],
                    "freqs": [peak["sub_f_hz"]],
                    "bins": [peak["sub_f_hz"] / bin_hz],
                    "contrasts": [peak["contrast_db"]],
                    "last_k": k,
                    "last_f": peak["sub_f_hz"],
                    "gap": 0
                }
                active_tracks.append(new_track)
                
    finished_tracks.extend(active_tracks)
    
    formatted_contours = []
    for trk in finished_tracks:
        L = len(trk["frames"])
        f_arr = np.array(trk["freqs"])
        c_arr = np.array(trk["contrasts"])
        
        formatted_contours.append({
            "contour_id": trk["track_id"],
            "filename": file_id,
            "channel": channel_idx,
            "frame_indices": trk["frames"],
            "times_s": trk["times"],
            "frequencies_hz": trk["freqs"],
            "bin_coordinates": trk["bins"],
            "contrast_db": trk["contrasts"],
            "n_points": L,
            "start_time_s": float(trk["times"][0]),
            "end_time_s": float(trk["times"][-1]),
            "duration_s": round(float(trk["times"][-1] - trk["times"][0]), 4),
            "start_freq_hz": round(float(f_arr[0]), 2),
            "end_freq_hz": round(float(f_arr[-1]), 2),
            "min_freq_hz": round(float(np.min(f_arr)), 2),
            "max_freq_hz": round(float(np.max(f_arr)), 2),
            "bandwidth_hz": round(float(np.max(f_arr) - np.min(f_arr)), 2),
            "mean_contrast_db": round(float(np.mean(c_arr)), 2),
            "input_transform": transform,
            "detector_status": "[ENGINEERING SUBSTITUTION]"
        })
        
    meta = {
        "file_id": file_id,
        "channel_idx": channel_idx,
        "detector_status": "[ENGINEERING SUBSTITUTION]",
        "detector_type": "classical_ridge_tracker_standin",
        "paper_specified_detector": "neural network-based whistle extraction [31] + MCA-CFAR post-processing [32],[33]",
        "substitution_reason": "References [31]-[33] are 2-page conference abstracts; weights/code unpublished",
        "input_transform": transform,
        "total_candidate_tracks": len(formatted_contours),
        "min_contrast_db": min_contrast_db,
        "max_gap_frames": max_gap_frames,
        "band_hz": [f_min, f_max]
    }
    return formatted_contours, meta
