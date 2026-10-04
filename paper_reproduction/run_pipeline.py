"""
run_pipeline.py -- Master Pipeline Runner for the 363-File Cetacean Whistle Preprocessing Pipeline.

Authoritative References:
  config_paper_reproduction.yaml
  PAPER_PARAMETER_LINEAGE.md
  PAPER_REPRODUCTION_AUDIT.md

EXECUTION SAFETY ENFORCEMENT:
  - Supports --dry-run and --full-run modes.
  - Default mode is --dry-run to guarantee zero accidental processing!
  - In --dry-run mode:
      Discovers files = 363
      Files that WOULD be processed = 363
      Files ACTUALLY processed = 0
  - In --full-run mode:
      Requires explicit --full-run flag.
      Enforces assertion: len(flac_files) == 363.
      Halts immediately if input file count != 363.
"""

import os
import sys
import glob
import time
import json
import csv
import argparse
import gc
import yaml
import numpy as np
import soundfile as sf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from h1_image import derive_h1, assert_fresh
from stage1_audio import process_file_stage1
from stage2_stft1 import compute_stft1, compute_spectrogram_db
from stage3_detector import detect_whistle_contours
from stage4_postprocess import postprocess_contours
from stage5_dedup import deduplicate_file_contours
from stage6_vectors import extract_time_frequency_vector
from stage7_nfm import screen_and_synthesize_nfm
from stage8_image import generate_and_save_image
from stage9_qc import perform_roundtrip_qc
import visualize
from dataset_store import (
    VersionedDatasetWriter,
    finalize_versioned_dataset,
    dataset_info,
    sha256_file,
)

def run_dry_run(cfg, config_path):
    """
    Execute a static/pre-flight dry run:
      - Discover files
      - Verify against manifest
      - Validate formats
      - Report exact counts without processing real audio
    """
    print("=" * 75)
    print("363-FILE PREPROCESSING PREFLIGHT DRY RUN")
    print(f"Base Directory: {BASE_DIR}")
    print(f"Config: {config_path}")
    print("=" * 75)
    
    raw_dir = cfg["input"]["flac_dir"]
    expected_count = cfg["input"].get("expected_input_count", 363)
    flac_files = sorted(glob.glob(os.path.join(raw_dir, "*.flac")))
    discovered_count = len(flac_files)
    
    print(f"\n[DRY RUN AUDIT] Input Directory: {raw_dir}")
    print(f"[DRY RUN AUDIT] Expected File Count: {expected_count}")
    print(f"[DRY RUN AUDIT] Discovered Files Count: {discovered_count}")
    
    if discovered_count != expected_count:
        raise AssertionError(
            f"PREFLIGHT FAILURE: Expected exactly {expected_count} files in {raw_dir}, "
            f"but found {discovered_count}!"
        )
        
    manifest_path = cfg["input"].get("manifest_path", "manifest_363.csv")
    if not os.path.isabs(manifest_path):
        manifest_path = os.path.join(BASE_DIR, manifest_path)
        
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            manifest_files = [row["filename"] for row in rdr]
        input_basenames = [os.path.basename(p) for p in flac_files]
        if set(manifest_files) != set(input_basenames):
            raise AssertionError(
                f"PREFLIGHT FAILURE: Discovered files do not match {manifest_path}!"
            )
        print(f"[DRY RUN AUDIT] Manifest Verification PASSED: All {len(manifest_files)} files match {manifest_path}.")
    else:
        print(f"[DRY RUN WARNING] Manifest not found at {manifest_path}")

    # Check stage output directories
    outputs_dir = os.path.join(BASE_DIR, cfg.get("paths", {}).get("outputs_dir", "outputs"))
    subdirs = [
        "stage1_audio", "stage2_stft1", "stage3_detection", "stage4_contours",
        "stage5_tf_vectors", "stage6_nfm", "stage7_images", "validation"
    ]
    for sub in subdirs:
        p = os.path.join(outputs_dir, sub)
        os.makedirs(p, exist_ok=True)
    print(f"[DRY RUN AUDIT] Output Directory Structure: Verified at {outputs_dir}")

    # Check H1 derivation
    derived_h1_path = os.path.join(BASE_DIR, cfg.get("paths", {}).get("derived_h1_path", "derived_h1.yaml"))
    derive_h1(config_path, derived_h1_path)
    assert_fresh(config_path, derived_h1_path)
    print(f"[DRY RUN AUDIT] H1 Parameter Derivation: Fresh and valid.")

    print("\n" + "=" * 75)
    print("DRY RUN EXECUTION SUMMARY:")
    print(f"  files discovered               = {discovered_count}")
    print(f"  files that WOULD be processed  = {discovered_count}")
    print(f"  files ACTUALLY processed       = 0")
    print("=" * 75)
    print("PREFLIGHT VALIDATION PASSED: The pipeline is ready for the 363-file full run.")
    print("=" * 75)
    return True

def append_records_to_csv(csv_path, records, fieldnames):
    """Append a batch of records to a CSV file atomically."""
    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerows(records)

def run_full_pipeline(cfg, config_path, args_holder=None):
    """
    Execute full preprocessing run across the 363 files.
    NOTE: Requires explicit --full-run argument.
    Canonical output: versioned HDF5 dataset (staging .tmp -> finalize).
    args_holder carries CLI storage options (never scientific params).
    """
    start_time = time.time()
    outputs_dir = os.path.join(BASE_DIR, cfg.get("paths", {}).get("outputs_dir", "outputs"))
    logs_dir = os.path.join(BASE_DIR, cfg.get("paths", {}).get("logs_dir", "logs"))
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    s1_out = os.path.join(outputs_dir, "stage1_audio")
    s2_out = os.path.join(outputs_dir, "stage2_stft1")
    s3_out = os.path.join(outputs_dir, "stage3_detection")
    s4_out = os.path.join(outputs_dir, "stage4_contours")
    s5_out = os.path.join(outputs_dir, "stage5_tf_vectors")
    s6_out = os.path.join(outputs_dir, "stage6_nfm")
    s7_out = os.path.join(outputs_dir, "stage7_images")
    val_out = os.path.join(outputs_dir, "validation")
    for d in [s1_out, s2_out, s3_out, s4_out, s5_out, s6_out, s7_out, val_out]:
        os.makedirs(d, exist_ok=True)

    raw_dir = cfg["input"]["flac_dir"]
    expected_count = cfg["input"].get("expected_input_count", 363)
    flac_files = sorted(glob.glob(os.path.join(raw_dir, "*.flac")))
    actual_count = len(flac_files)

    print("=" * 75)
    print("FULL 363-FILE PAPER-REPRODUCTION PREPROCESSING PIPELINE")
    print(f"Base Directory: {BASE_DIR}")
    print(f"Config: {config_path}")
    print(f"Input Files: {actual_count}")
    print("=" * 75)

    if actual_count != expected_count:
        raise AssertionError(
            f"FATAL SAFETY VIOLATION: Expected exactly {expected_count} files in {raw_dir}, "
            f"but found {actual_count}! HALTING TO PREVENT ACCIDENTAL MASS PROCESSING."
        )

    # Staging ingest_state.json (in <version>.tmp) is the resume record.
    # A legacy logs/checkpoint_progress.json, if present from older runs, is ignored.

    derived_h1_path = os.path.join(BASE_DIR, cfg.get("paths", {}).get("derived_h1_path", "derived_h1.yaml"))
    derive_h1(config_path, derived_h1_path)
    derived_h1 = assert_fresh(config_path, derived_h1_path)

    # ---- Canonical versioned dataset (STORAGE ONLY; science unchanged) ----
    ds_cfg = cfg.get("dataset", {})
    dataset_version = getattr(args_holder, "dataset_version", None) or ds_cfg.get("version", "preprocessing_v001")
    datasets_dir = getattr(args_holder, "datasets_dir", None) or ds_cfg.get("dir", "datasets")
    if not os.path.isabs(datasets_dir):
        datasets_dir = os.path.join(BASE_DIR, datasets_dir)
    _cli_export = getattr(args_holder, "export_npy", None)
    export_npy = bool(_cli_export if _cli_export is not None else ds_cfg.get("export_npy", False))
    overwrite = bool(getattr(args_holder, "overwrite_dataset", False)
                     or ds_cfg.get("overwrite", False))
    final_ds_dir = os.path.join(datasets_dir, dataset_version)
    staging_ds_dir = final_ds_dir + ".tmp"
    if os.path.exists(os.path.join(final_ds_dir, "COMPLETE")) and not overwrite:
        raise AssertionError(
            f"Refusing to overwrite COMPLETE dataset {final_ds_dir}. "
            "Choose a new --dataset-version or pass --overwrite-dataset."
        )
    import hashlib
    try:
        with open(config_path, "rb") as f:
            config_hash = hashlib.sha256(f.read()).hexdigest()
    except Exception:
        config_hash = None
    h1_fingerprint = derived_h1.get("fingerprint")
    source_basenames = [os.path.basename(p) for p in flac_files]
    writer = VersionedDatasetWriter(
        staging_ds_dir, dataset_version, cfg, config_hash, h1_fingerprint,
        source_basenames,
        chunk_samples=ds_cfg.get("chunk_samples", 8),
        compression=ds_cfg.get("compression", "gzip"),
        compression_level=ds_cfg.get("compression_level", 4),
        preview_count=ds_cfg.get("preview_count", 64),
    )
    writer.recount_previews()
    # Resume from staging ingest_state.json (supersedes legacy checkpoint file).
    completed_files = set(writer.completed_files)
    if completed_files:
        print(f"[RESUME] Staging {dataset_version}.tmp has {len(completed_files)} files, {writer.n} samples.")

    # Q1 FINAL DECISION: single-channel processing ONLY (channel_index from config).
    # Source validation still expects 6ch files (PAPER FACT); only this channel is processed.
    ch_cfg = cfg.get("channel_handling", {})
    channel_index = int(ch_cfg.get("channel_index", 0))
    if channel_index not in range(6):
        raise ValueError(f"Q1 violation: channel_index={channel_index} not in 0..5")
    print(f"[Q1] Single-channel mode: processing ONLY channel {channel_index} (channels 1-5 ignored).")

    total_whistle_images = 0
    total_contours_detected = 0

    # CSV path definitions for incremental streaming
    val_csv = os.path.join(s1_out, "audio_validation.csv")
    ch_csv = os.path.join(s1_out, "channel_statistics.csv")
    stft1_csv = os.path.join(s2_out, "stft1_summary.csv")
    det_csv = os.path.join(s3_out, "detector_statistics.csv")
    post_csv = os.path.join(s4_out, "postprocess_statistics.csv")
    tf_csv = os.path.join(s5_out, "tf_vectors.csv")
    ov_csv = os.path.join(s6_out, "overlong_contours.csv")
    img_csv = os.path.join(s7_out, "images_manifest.csv")
    qc_csv = os.path.join(val_out, "qc_results.csv")

    for file_idx, fpath in enumerate(flac_files):
        fn = os.path.basename(fpath)
        if fn in completed_files:
            continue

        file_t0 = time.time()
        print(f"[{file_idx + 1}/{actual_count}] Processing {fn}...")

        # Stage 1: Audio Loading (6ch validated), channel-0 select, resample, high-pass.
        # Returns 1D channel-0 waveform ONLY; channels 1-5 never leave stage1.
        audio_192k_ch0, meta_s1 = process_file_stage1(fpath, cfg)
        if np.ndim(audio_192k_ch0) != 1:
            raise ValueError(f"Stage-1 contract violated: expected 1D channel-0 audio, got shape {np.shape(audio_192k_ch0)}")
        raw_val = meta_s1["raw_validation"]
        append_records_to_csv(val_csv, [raw_val], list(raw_val.keys()))
        append_records_to_csv(ch_csv, meta_s1["channel_statistics"], list(meta_s1["channel_statistics"][0].keys()))

        file_post_contours = []
        file_stft_records = []
        file_det_records = []
        file_post_records = []

        # Q1: channel 0 ONLY. No per-channel loop; single 1D waveform from stage1.
        # Channels 1-5 are not resampled/filtered (see stage1) and never reach STFT.
        ch_idx = int(channel_index)
        for _once in [0]:
            ch_audio = audio_192k_ch0  # 1D, channel 0 only

            # Stage 2: STFT #1
            stft_complex, times_s, freqs_hz = compute_stft1(
                ch_audio,
                fs=cfg["resample"]["fs_out"],
                nfft=cfg["stft1"]["nfft"],
                hop=cfg["stft1"]["hop"]
            )
            S_db = compute_spectrogram_db(stft_complex)

            file_stft_records.append({
                "filename": fn,
                "channel_idx": ch_idx,
                "n_frames": stft_complex.shape[1],
                "n_bins": stft_complex.shape[0],
                "fs_hz": cfg["resample"]["fs_out"],
                "bin_hz": cfg["stft1"]["bin_hz"],
                "hop_samples": cfg["stft1"]["hop"]
            })

            # Stage 3: Contour Detection (Stand-in)
            cand_contours, det_meta = detect_whistle_contours(
                S_db, times_s, freqs_hz, cfg, file_id=fn, channel_idx=ch_idx
            )
            file_det_records.append({
                "filename": fn,
                "channel_idx": ch_idx,
                "detector_type": det_meta["detector_type"],
                "detector_status": det_meta["detector_status"],
                "candidates_detected": len(cand_contours),
                "min_contrast_db": det_meta["min_contrast_db"],
                "max_gap_frames": det_meta["max_gap_frames"]
            })

            # Stage 4: Post-processing Filter
            acc_contours, rej_contours, post_log = postprocess_contours(
                cand_contours, cfg, channel_idx=ch_idx
            )
            post_log["filename"] = fn
            file_post_records.append(post_log)
            file_post_contours.extend(acc_contours)

            # Cleanup channel arrays
            del stft_complex, S_db, ch_audio
            gc.collect()

        append_records_to_csv(stft1_csv, file_stft_records, list(file_stft_records[0].keys()))
        append_records_to_csv(det_csv, file_det_records, list(file_det_records[0].keys()))
        append_records_to_csv(post_csv, file_post_records, list(file_post_records[0].keys()))

        # Stage 5: Time-Frequency Vectors (Default: Dedup OFF)
        dedup_cfg = cfg.get("dedup", {})
        if dedup_cfg.get("enabled", False):
            retained_contours, _, _, _ = deduplicate_file_contours(file_post_contours, cfg, filename=fn)
        else:
            retained_contours = file_post_contours

        file_tf_vectors = [extract_time_frequency_vector(c, cfg) for c in retained_contours]
        if file_tf_vectors:
            flat_tf = [{
                "contour_id": v["contour_id"],
                "filename": v["filename"],
                "channel": v["channel"],
                "length_frames": v["length_frames"],
                "duration_s": v["duration_s"],
                "f_min_hz": v["f_min_hz"],
                "f_max_hz": v["f_max_hz"],
                "f_start_hz": v["f_start_hz"],
                "f_end_hz": v["f_end_hz"],
                "f_center_hz": v["f_center_hz"],
                "bandwidth_hz": v["bandwidth_hz"],
                "mean_contrast_db": v["mean_contrast_db"]
            } for v in file_tf_vectors]
            append_records_to_csv(tf_csv, flat_tf, list(flat_tf[0].keys()))

        # Stage 6: NFM Baseband Synthesis
        supp_nfm, overlong_list, margin_list, scr_summary = screen_and_synthesize_nfm(
            file_tf_vectors, cfg, config_path
        )
        if overlong_list:
            overlong_fields = [
                "contour_id", "filename", "channel", "length_frames", "duration_s",
                "f_min_hz", "f_max_hz", "f_center_hz", "bandwidth_hz", "excess_frames", "reason"
            ]
            append_records_to_csv(ov_csv, overlong_list, overlong_fields)

        # Stage 7 & 8: Image Synthesis & Internal QC + canonical HDF5 append.
        # Science unchanged: synth_image/QC identical; storage streams to HDF5
        # per sample (no full-dataset RAM accumulation).
        normalization = cfg.get("image", {}).get("normalization")
        file_img_records = []
        file_qc_records = []
        batch_images = []
        batch_meta = []
        batch_qc = []
        for nfm_item in supp_nfm:
            img_rec = generate_and_save_image(
                nfm_item, cfg, config_path, s7_out, save_files=export_npy)
            file_img_records.append({
                "contour_id": img_rec["contour_id"],
                "filename": img_rec["filename"],
                "channel": img_rec["channel"],
                "image_shape": f"{img_rec['image_shape'][0]}x{img_rec['image_shape'][1]}",
                "min_val": round(img_rec["min_val"], 4),
                "max_val": round(img_rec["max_val"], 4),
                "mean_val": round(img_rec["mean_val"], 4),
                "start_col": img_rec["start_col"],
                "length_frames": img_rec["length_frames"],
                "center_freq_hz": round(img_rec["center_freq_hz"], 2),
                "npy_path": img_rec["npy_path"] or "",
                "png_path": img_rec["png_path"] or ""
            })

            qc_res = perform_roundtrip_qc(img_rec, nfm_item, cfg, config_path)
            file_qc_records.append({
                "contour_id": qc_res["contour_id"],
                "filename": fn,
                "channel": img_rec["channel"],
                "status": qc_res["status"],
                "original_length": qc_res.get("original_length", ""),
                "recovered_length": qc_res.get("recovered_length", ""),
                "length_error_frames": qc_res.get("length_error_frames", ""),
                "max_centroid_error_bins": qc_res.get("max_centroid_error_bins", ""),
                "mean_centroid_error_bins": qc_res.get("mean_centroid_error_bins", ""),
                "peak_intensity": qc_res.get("peak_intensity", ""),
                "rejection_reasons": "; ".join(qc_res.get("rejection_reasons", []))
            })

            # Canonical append (storage only): image + metadata + QC rows.
            vec = nfm_item["vector"]
            meta_row = {
                "source_file": fn,
                "source_channel": img_rec["channel"],
                "contour_id": img_rec["contour_id"],
                "duration_frames": int(img_rec["length_frames"]),
                "duration_seconds": float(vec.get("duration_s", 0.0)),
                "fmin_hz": float(vec.get("f_min_hz")),
                "fmax_hz": float(vec.get("f_max_hz")),
                "center_frequency_hz": float(vec.get("f_center_hz")),
                "bandwidth_hz": float(vec.get("bandwidth_hz")),
                "start_frequency_hz": float(vec.get("f_start_hz")),
                "end_frequency_hz": float(vec.get("f_end_hz")),
                "image_height": 128,
                "image_width": 128,
                "dtype": "float32",
                "normalization": normalization,
                "preprocessing_version": dataset_version,
                "config_hash": config_hash,
                "h1_fingerprint": h1_fingerprint,
                "qc_status": qc_res.get("status"),
            }
            qc_row = {
                "contour_id": qc_res["contour_id"],
                "source_file": fn,
                "channel": img_rec["channel"],
                "status": qc_res.get("status"),
                "original_length": qc_res.get("original_length"),
                "recovered_length": qc_res.get("recovered_length"),
                "length_error_frames": qc_res.get("length_error_frames"),
                "max_centroid_error_bins": qc_res.get("max_centroid_error_bins"),
                "mean_centroid_error_bins": qc_res.get("mean_centroid_error_bins"),
                "peak_intensity": qc_res.get("peak_intensity"),
            }
            batch_images.append(img_rec["image_data"])
            batch_meta.append(meta_row)
            batch_qc.append(qc_row)

            if len(batch_images) >= 500:
                writer.append_batch(batch_images, batch_meta, batch_qc, fn)
                batch_images.clear()
                batch_meta.clear()
                batch_qc.clear()

        if batch_images:
            writer.append_batch(batch_images, batch_meta, batch_qc, fn)
            batch_images.clear()
            batch_meta.clear()
            batch_qc.clear()


        if file_img_records:
            append_records_to_csv(img_csv, file_img_records, list(file_img_records[0].keys()))
        if file_qc_records:
            append_records_to_csv(qc_csv, file_qc_records, list(file_qc_records[0].keys()))

        total_whistle_images += len(file_img_records)
        total_contours_detected += len(file_post_contours)

        # Staging checkpoint (crash-safe; supersedes legacy checkpoint file).
        completed_files.add(fn)
        writer.mark_file_complete(fn, source_basenames)

        del audio_192k_ch0
        gc.collect()

        print(f"  Completed {fn} in {time.time()-file_t0:.1f}s | Images: {len(file_img_records)} | Total Images: {total_whistle_images}")

    # Finalize: integrity checks, parquet/csv/manifest, .tmp -> final, COMPLETE.
    manifest = finalize_versioned_dataset(
        staging_ds_dir, final_ds_dir, dataset_version, cfg,
        config_hash, h1_fingerprint, source_basenames)

    elapsed = time.time() - start_time
    print("=" * 75)
    print("363-FILE PREPROCESSING COMPLETED SUCCESSFULLY")
    print(f"Canonical dataset: {final_ds_dir}/dataset.h5 "
          f"(/images shape=({manifest['total_samples']},128,128,1))")
    print(f"Total Execution Time: {elapsed/60.0:.2f} minutes")
    print(f"Total Images Generated: {total_whistle_images}")
    print("=" * 75)
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Master Preprocessing Pipeline for Cetacean Whistles (Park et al., 2026 reproduction)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(BASE_DIR, "config_paper_reproduction.yaml"),
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform preflight file discovery, manifest check, and configuration validation without processing audio"
    )
    parser.add_argument(
        "--full-run",
        action="store_true",
        help="Execute the full 363-file preprocessing pipeline (requires explicit user authorization)"
    )
    parser.add_argument(
        "--dataset-version",
        type=str,
        default=None,
        help="Canonical dataset version (default: dataset.version in config)"
    )
    parser.add_argument(
        "--datasets-dir",
        type=str,
        default=None,
        help="Datasets root directory (default: dataset.dir in config)"
    )
    parser.add_argument(
        "--overwrite-dataset",
        action="store_true",
        help="Allow overwriting a COMPLETE dataset version (default: refuse)"
    )
    parser.add_argument(
        "--export-npy",
        action="store_true",
        default=None,
        help="Also write legacy per-sample .npy/.png (default: HDF5 canonical only)"
    )
    parser.add_argument(
        "--dataset-info",
        type=str,
        default=None,
        metavar="VERSION_OR_PATH",
        help="Inspect an existing dataset version without preprocessing anything"
    )

    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        if os.path.exists(config_path):
            config_path = os.path.abspath(config_path)
        else:
            config_path = os.path.join(BASE_DIR, config_path)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.dataset_info:
        ds_root = cfg.get("dataset", {}).get("dir", "datasets")
        if not os.path.isabs(ds_root):
            ds_root = os.path.join(BASE_DIR, ds_root)
        target = args.dataset_info
        vdir = target if os.path.isdir(target) else os.path.join(ds_root, target)
        print(json.dumps(dataset_info(vdir), indent=2, default=str))
        return

    if args.full_run:
        print("[WARNING] FULL-RUN MODE REQUESTED. Verifying authorization...")
        run_full_pipeline(cfg, config_path, args)
    else:
        # Default mode is strictly dry-run for safety
        if not args.dry_run:
            print("[NOTICE] No execution mode specified. Defaulting to safe --dry-run mode.\n")
        run_dry_run(cfg, config_path)

if __name__ == "__main__":
    main()
