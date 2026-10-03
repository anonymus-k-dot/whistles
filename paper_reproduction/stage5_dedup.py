"""
stage5_dedup.py -- Cross-Channel Contour Deduplication (Optional Engineering Ablation).

Authoritative Reference:
  PREPROCESSING_PIPELINE.md (Section 1-3, D7, D8, D20)
  config_paper_test.yaml (block: dedup)

Lineage Categories:
  - Cross-channel deduplication: [NOT SPECIFIED BY PAPER] / H1.
      Q1 FINAL DECISION: single-channel-0 mode makes dedup NOT APPLICABLE.
      DEFAULT PAPER REPRODUCTION: deduplication is OFF (enabled: false).
      Retained strictly as an optional engineering sensitivity ablation.
"""

import os
import sys
import hashlib
import numpy as np

def stable_hash_str(s):
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:8], 16)

def deduplicate_file_contours(file_contours, cfg, filename=""):
    """
    Deduplicate multi-channel whistle contours within one file.
    If dedup is disabled (default in paper reproduction), returns all contours unchanged.
    """
    dedup_cfg = cfg.get("dedup", {})
    enabled = dedup_cfg.get("enabled", False) # DEFAULT REPRODUCTION: FALSE
    
    if not enabled or len(file_contours) <= 1:
        return file_contours, [], [], {
            "enabled": enabled,
            "total_before": len(file_contours),
            "total_after": len(file_contours),
            "duplicates_removed": 0,
            "ambiguous_groups": 0
        }
        
    max_lag_s = dedup_cfg.get("max_lag_s", 0.010)
    min_overlap = dedup_cfg.get("min_temporal_overlap_fraction", 0.5)
    max_freq_diff_hz = dedup_cfg.get("max_mean_abs_freq_diff_hz", 93.75)
    keep_policy = dedup_cfg.get("keep", "seeded_random")
    
    root_seed = cfg.get("run", {}).get("seed", 0)
    file_seed = stable_hash_str(filename)
    ss = np.random.SeedSequence([root_seed, file_seed])
    rng = np.random.default_rng(ss)
    
    file_contours = sorted(file_contours, key=lambda c: c["start_time_s"])
    n = len(file_contours)
    adj = {i: set([i]) for i in range(n)}
    pair_metrics = {}
    
    for i in range(n):
        c1 = file_contours[i]
        ch1 = c1["channel"]
        t1_st, t1_end = c1["start_time_s"], c1["end_time_s"]
        dur1 = c1["duration_s"]
        f1_map = dict(zip(c1["frame_indices"], c1["frequencies_hz"]))
        
        for j in range(i + 1, n):
            c2 = file_contours[j]
            t2_st, t2_end = c2["start_time_s"], c2["end_time_s"]
            dur2 = c2["duration_s"]
            
            lag = t2_st - t1_st
            if lag > max_lag_s:
                break
                
            ch2 = c2["channel"]
            if ch1 == ch2:
                continue
                
            overlap_st = max(t1_st, t2_st)
            overlap_end = min(t1_end, t2_end)
            overlap_dur = overlap_end - overlap_st
            min_dur = min(dur1, dur2)
            
            if min_dur <= 0 or (overlap_dur / min_dur) < min_overlap:
                continue
                
            f2_map = dict(zip(c2["frame_indices"], c2["frequencies_hz"]))
            common_frames = sorted(list(set(f1_map.keys()).intersection(set(f2_map.keys()))))
            if len(common_frames) == 0:
                continue
                
            mean_df = float(np.mean([abs(f1_map[k] - f2_map[k]) for k in common_frames]))
            if mean_df <= max_freq_diff_hz:
                adj[i].add(j)
                adj[j].add(i)
                pair_metrics[(i, j)] = {
                    "lag_s": round(lag, 6),
                    "temporal_overlap_frac": round(overlap_dur / min_dur, 4),
                    "mean_freq_diff_hz": round(mean_df, 2)
                }
                
    visited = set()
    components = []
    for i in range(n):
        if i not in visited:
            group = []
            queue = [i]
            visited.add(i)
            while queue:
                curr = queue.pop(0)
                group.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            components.append(group)
            
    retained_contours = []
    discarded_contours = []
    groups_record = []
    group_id_counter = 0
    
    for grp_indices in components:
        if len(grp_indices) == 1:
            retained_contours.append(file_contours[grp_indices[0]])
            continue
            
        group_id_counter += 1
        group_members = [file_contours[idx] for idx in grp_indices]
        channels_in_group = [m["channel"] for m in group_members]
        
        if keep_policy == "seeded_random":
            chosen_local_idx = int(rng.integers(0, len(group_members)))
            retained = group_members[chosen_local_idx]
        else:
            retained = group_members[0]
            
        retained_contours.append(retained)
        
        discarded_ids = []
        for m in group_members:
            if m["contour_id"] != retained["contour_id"]:
                c_disc = dict(m)
                c_disc["discard_reason"] = "cross_channel_duplicate"
                c_disc["merged_into"] = retained["contour_id"]
                discarded_contours.append(c_disc)
                discarded_ids.append(m["contour_id"])
                
        groups_record.append({
            "group_id": f"{filename}_grp_{group_id_counter:03d}",
            "status": "deduplicated",
            "participating_channels": channels_in_group,
            "retained_id": retained["contour_id"],
            "discarded_ids": discarded_ids
        })
        
    summary = {
        "enabled": enabled,
        "total_before": len(file_contours),
        "total_after": len(retained_contours),
        "duplicates_removed": len(discarded_contours),
        "ambiguous_groups": 0,
        "match_groups_count": len(groups_record)
    }
    return retained_contours, discarded_contours, groups_record, summary
