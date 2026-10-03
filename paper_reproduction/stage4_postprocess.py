"""
stage4_postprocess.py -- Stage 4: Contour Post-Processing.

Authoritative Reference:
  PREPROCESSING_PIPELINE.md (Section 1-3, D6)
  config_paper_test.yaml (block: detector)

Lineage Categories:
  - Paper post-processing: [PAPER-SPECIFIED]
      Paper Section II.C states:
      "The extraction process was combined with a post-processing algorithm [32], [33]
       that converts detection outputs into individual whistle tracks."
  - Duration filter (Q3): 10 frames = H1 (paper silent).
  - Harmonic rejection (Q3): disabled = H1 (paper silent).
      Harmonic rejection was a Phase-1 engineering choice; DISABLED by default in
      the paper-reproduction pipeline.
"""

import os
import sys
import numpy as np

def postprocess_contours(candidate_contours, cfg, channel_idx=None):
    """
    Apply post-processing rules to candidate contours.
    
    Returns:
      accepted_contours: List of contours passing filters
      rejected_contours: List of rejected contours with reasons
      log_record: Dict with counts and rejection breakdown
    """
    post_cfg = cfg.get("postprocess", cfg.get("detector", {}))
    min_dur = post_cfg.get("min_duration_frames", 10)
    reject_harmonics = post_cfg.get("reject_harmonics", False)
    
    total_in = len(candidate_contours)
    rejected_list = []
    
    # 1. Minimum duration filter (Implementation Required for stand-in)
    survived_duration = []
    dur_rejected = 0
    for c in candidate_contours:
        if c["n_points"] < min_dur:
            dur_rejected += 1
            c_rej = dict(c)
            c_rej["rejection_reason"] = "duration_too_short"
            c_rej["rejection_detail"] = f"Length {c['n_points']} < min_duration_frames ({min_dur})"
            rejected_list.append(c_rej)
        else:
            survived_duration.append(c)
            
    # 2. Harmonic rejection (Optional ablation; default is disabled in paper reproduction)
    survived_harmonic = []
    harmonic_rejected_ids = set()
    harmonic_rejections = {}
    
    if reject_harmonics and len(survived_duration) > 1:
        sorted_by_snr = sorted(survived_duration, key=lambda x: x["mean_contrast_db"], reverse=True)
        for i in range(len(sorted_by_snr)):
            c_strong = sorted_by_snr[i]
            if c_strong["contour_id"] in harmonic_rejected_ids:
                continue
            t_strong_start = c_strong["start_time_s"]
            t_strong_end = c_strong["end_time_s"]
            
            for j in range(i + 1, len(sorted_by_snr)):
                c_weak = sorted_by_snr[j]
                if c_weak["contour_id"] in harmonic_rejected_ids:
                    continue
                t_weak_start = c_weak["start_time_s"]
                t_weak_end = c_weak["end_time_s"]
                
                overlap_start = max(t_strong_start, t_weak_start)
                overlap_end = min(t_strong_end, t_weak_end)
                overlap_dur = overlap_end - overlap_start
                min_dur_pair = min(c_strong["duration_s"], c_weak["duration_s"])
                
                if min_dur_pair > 0 and (overlap_dur / min_dur_pair) >= 0.5:
                    frames_strong = set(c_strong["frame_indices"])
                    frames_weak = set(c_weak["frame_indices"])
                    common_frames = sorted(list(frames_strong.intersection(frames_weak)))
                    
                    if len(common_frames) >= 5:
                        f_strong_map = dict(zip(c_strong["frame_indices"], c_strong["frequencies_hz"]))
                        f_weak_map = dict(zip(c_weak["frame_indices"], c_weak["frequencies_hz"]))
                        ratios = [f_weak_map[k] / (f_strong_map[k] + 1e-6) for k in common_frames]
                        med_ratio = float(np.median(ratios))
                        
                        if (1.85 <= med_ratio <= 2.15 or 2.80 <= med_ratio <= 3.20) and (c_weak["mean_contrast_db"] <= c_strong["mean_contrast_db"]):
                            harmonic_rejected_ids.add(c_weak["contour_id"])
                            harmonic_rejections[c_weak["contour_id"]] = {
                                "reason": "harmonic",
                                "detail": f"Harmonic ratio {med_ratio:.2f} relative to stronger track {c_strong['contour_id']}"
                            }
        for c in survived_duration:
            if c["contour_id"] in harmonic_rejected_ids:
                c_rej = dict(c)
                c_rej["rejection_reason"] = harmonic_rejections[c["contour_id"]]["reason"]
                c_rej["rejection_detail"] = harmonic_rejections[c["contour_id"]]["detail"]
                rejected_list.append(c_rej)
            else:
                survived_harmonic.append(c)
    else:
        survived_harmonic = survived_duration
        
    accepted_contours = survived_harmonic
    
    log_record = {
        "channel_idx": channel_idx,
        "input_contour_count": total_in,
        "accepted_contour_count": len(accepted_contours),
        "rejected_contour_count": len(rejected_list),
        "rejected_by_duration": dur_rejected,
        "rejected_by_harmonic": len(harmonic_rejected_ids),
        "harmonic_rejection_enabled": reject_harmonics
    }
    return accepted_contours, rejected_list, log_record
