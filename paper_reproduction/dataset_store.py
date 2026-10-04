"""
dataset_store.py -- Versioned canonical preprocessing dataset (STORAGE ONLY).

Engineering storage decision, NOT a paper claim. The paper does not specify
HDF5/parquet/versioning; this module only changes HOW validated scientific
outputs are stored, never the scientific procedure (Q1-Q6 untouched).

Layout:
    datasets/<version>.tmp/   staging while ingesting (INCOMPLETE)
    datasets/<version>/       final after finalize() (COMPLETE)

Final contents:
    dataset.h5       /images (N,128,128,1) float32 [0,1], chunked, gzip
    metadata.parquet one row per sample (see METADATA_COLUMNS)
    manifest.json    provenance (config hash, H1 fingerprint, software, params)
    qc.csv           per-sample QC rows
    previews/        first-N PNG visualizations (debug only, NOT training data)
    README.md        dataset documentation
    COMPLETE         marker (only after integrity checks pass)

Memory discipline: images stream into HDF5 per sample (resize-append);
only small metadata/QC row dicts accumulate (via crash-safe JSONL sidecars).
"""

import os
import json
import glob
import hashlib
import shutil
import datetime

import numpy as np
import h5py
import pandas as pd

IMAGE_SHAPE = (128, 128, 1)
IMAGE_DTYPE = np.float32

METADATA_COLUMNS = [
    "image_id",
    "dataset_index",
    "source_file",
    "source_channel",
    "contour_id",
    "duration_frames",
    "duration_seconds",
    "fmin_hz",
    "fmax_hz",
    "center_frequency_hz",
    "bandwidth_hz",
    "start_frequency_hz",
    "end_frequency_hz",
    "image_height",
    "image_width",
    "dtype",
    "normalization",
    "preprocessing_version",
    "config_hash",
    "h1_fingerprint",
    "qc_status",
]

QC_COLUMNS = [
    "image_id",
    "dataset_index",
    "contour_id",
    "source_file",
    "channel",
    "status",
    "original_length",
    "recovered_length",
    "length_error_frames",
    "max_centroid_error_bins",
    "mean_centroid_error_bins",
    "peak_intensity",
]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _utcnow():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_commit():
    try:
        import subprocess
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        c = (out.stdout or "").strip()
        return c if c else None
    except Exception:
        return None


def _software_versions():
    vers = {}
    for mod in ("python", "numpy", "scipy", "h5py", "torch", "pandas", "pyarrow", "soundfile"):
        try:
            if mod == "python":
                import sys
                vers[mod] = sys.version.split()[0]
            else:
                m = __import__(mod)
                vers[mod] = getattr(m, "__version__", "installed")
        except Exception:
            vers[mod] = None
    return vers


class VersionedDatasetWriter:
    """Appendable writer for one preprocessing dataset version (staging dir)."""

    def __init__(self, staging_dir, version, cfg, config_hash, h1_fingerprint,
                 source_files, chunk_samples=8, compression="gzip",
                 compression_level=4, preview_count=64):
        self.staging_dir = staging_dir
        self.version = version
        self.cfg = cfg
        self.config_hash = config_hash
        self.h1_fingerprint = h1_fingerprint
        self.chunk_samples = int(chunk_samples)
        self.compression = compression
        self.compression_level = int(compression_level)
        self.preview_count = int(preview_count)
        self.n = 0
        self.previews_written = 0

        os.makedirs(staging_dir, exist_ok=True)
        previews_dir = os.path.join(staging_dir, "previews")
        os.makedirs(previews_dir, exist_ok=True)

        self.h5_path = os.path.join(staging_dir, "dataset.h5")
        self.meta_jsonl = os.path.join(staging_dir, "metadata_rows.jsonl")
        self.qc_jsonl = os.path.join(staging_dir, "qc_rows.jsonl")
        self.state_path = os.path.join(staging_dir, "ingest_state.json")

        if os.path.exists(self.h5_path):
            # Resume: recover count from existing staging dataset.
            with h5py.File(self.h5_path, "r") as f:
                self.n = int(f["/images"].shape[0])
            self._load_state_rows_count()
        else:
            with h5py.File(self.h5_path, "w") as f:
                f.create_dataset(
                    "/images",
                    shape=(0, 128, 128, 1),
                    maxshape=(None, 128, 128, 1),
                    dtype="float32",
                    chunks=(self.chunk_samples, 128, 128, 1),
                    compression=self.compression,
                    compression_opts=self.compression_level,
                    shuffle=True,
                )
                f["/images"].attrs["storage_note"] = (
                    "Engineering storage (HDF5). NOT specified by the paper."
                )
                f["/images"].attrs["value_range"] = "[0,1]"
            open(self.meta_jsonl, "a").close()
            open(self.qc_jsonl, "a").close()

        self.completed_files = self._read_completed_files()
        # source_files recorded for manifest; also persist immediately.
        self._write_state(source_files)

    # -- resume helpers ----------------------------------------------------
    def _load_state_rows_count(self):
        # JSONL sidecars are the source of truth for rows; nothing to load
        # into RAM (rows stay on disk until finalize).
        pass

    def _read_completed_files(self):
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, "r", encoding="utf-8") as f:
                    return set(json.load(f).get("completed_files", []))
            except Exception:
                return set()
        return set()

    def _write_state(self, source_files):
        state = {
            "version": self.version,
            "status": "INGEST_IN_PROGRESS",
            "completed_files": sorted(self.completed_files),
            "total_samples": self.n,
            "config_hash": self.config_hash,
            "h1_fingerprint": self.h1_fingerprint,
            "source_file_count": len(source_files),
        }
        tmp = self.state_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, self.state_path)

    # -- ingest ------------------------------------------------------------
    def append_batch(self, images_2d, meta_rows, qc_rows, source_file):
        """Append a batch of validated 128x128 float32 images + their rows in a single I/O operation.

        images_2d: list of (128,128) or (128,128,1) float32 in [0,1], or an (N,128,128,1) array.
        Only shape/dtype/finiteness are checked here (storage invariants); science untouched.
        """
        if not images_2d:
            return []
        
        arr_list = []
        for img in images_2d:
            arr = np.asarray(img, dtype=np.float32)
            if arr.shape == (128, 128):
                arr = arr[:, :, None]
            if arr.shape != (128, 128, 1):
                raise ValueError(f"Bad image shape {arr.shape}, expected (128,128,1)")
            if arr.dtype != np.float32:
                raise ValueError(f"Bad image dtype {arr.dtype}")
            if not np.all(np.isfinite(arr)):
                raise ValueError("Non-finite image values")
            if float(np.min(arr)) < -1e-6 or float(np.max(arr)) > 1.0 + 1e-6:
                raise ValueError(
                    f"Image outside [0,1]: min={np.min(arr)} max={np.max(arr)}"
                )
            arr_list.append(arr)

        n_batch = len(arr_list)
        batch_arr = np.stack(arr_list, axis=0)  # shape (n_batch, 128, 128, 1)
        start_idx = self.n
        end_idx = start_idx + n_batch

        with h5py.File(self.h5_path, "a") as f:
            d = f["/images"]
            d.resize((end_idx, 128, 128, 1))
            d[start_idx:end_idx] = batch_arr

        meta_lines = []
        qc_lines = []
        res = []
        for i in range(n_batch):
            idx = start_idx + i
            image_id = f"{self.version}-{idx:07d}"
            meta_row = dict(meta_rows[i])
            meta_row["image_id"] = image_id
            meta_row["dataset_index"] = idx
            qc_row = dict(qc_rows[i])
            qc_row["image_id"] = image_id
            qc_row["dataset_index"] = idx

            meta_lines.append(json.dumps(meta_row, default=str))
            qc_lines.append(json.dumps(qc_row, default=str))

            if self.previews_written < self.preview_count:
                self._write_preview(arr_list[i][:, :, 0], idx)
                self.previews_written += 1

            res.append((image_id, idx))

        with open(self.meta_jsonl, "a", encoding="utf-8") as f:
            f.write("\n".join(meta_lines) + "\n")
        with open(self.qc_jsonl, "a", encoding="utf-8") as f:
            f.write("\n".join(qc_lines) + "\n")

        self.n = end_idx
        return res

    def append_sample(self, image_2d, meta_row, qc_row, source_file):
        """Append one validated 128x128 float32 image + its rows."""
        res = self.append_batch([image_2d], [meta_row], [qc_row], source_file)
        return res[0]


    def _write_preview(self, img_hw, idx):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            u8 = np.clip(img_hw * 255.0, 0, 255).astype(np.uint8)
            p = os.path.join(self.staging_dir, "previews", f"sample_{idx:07d}.png")
            plt.imsave(p, u8, cmap="gray", vmin=0, vmax=255)
        except Exception:
            pass  # previews are debug-only; never fail ingestion

    def mark_file_complete(self, filename, source_files):
        self.completed_files.add(filename)
        self._write_state(source_files)

    def recount_previews(self):
        pats = glob.glob(os.path.join(self.staging_dir, "previews", "*.png"))
        self.previews_written = len(pats)


def _read_jsonl(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_manifest(version, cfg, config_hash, h1_fingerprint, source_files,
                   total_samples, creation_timestamp):
    det = cfg.get("detector", {})
    image = cfg.get("image", {})
    nfm = cfg.get("nfm", {})
    val = cfg.get("validation", {})
    return {
        "dataset_version": version,
        "status": "COMPLETE",
        "creation_timestamp": creation_timestamp,
        "source_dataset": "NOAA DCLDE 2022 Cruise 1705 Encounter 191",
        "source_file_count": len(source_files),
        "source_files": sorted(source_files),
        "total_samples": int(total_samples),
        "source_sampling_rate": cfg.get("input", {}).get("expected_fs"),
        "target_sampling_rate": cfg.get("resample", {}).get("fs_out"),
        "channel_index": cfg.get("channel_handling", {}).get("channel_index", 0),
        "channel_note": "PAPER FACT: reproduction uses channel 0 only.",
        "preprocessing_config_snapshot": cfg,
        "config_hash": config_hash,
        "h1_fingerprint": h1_fingerprint,
        "git_commit": _git_commit(),
        "repository_version": cfg.get("version", None),
        "image_shape": [128, 128, 1],
        "image_dtype": "float32",
        "normalization_method": image.get("normalization"),
        "normalization_note": "H1 engineering storage detail; paper specifies [0,1] range only.",
        "qc_thresholds": {
            "contour_decode_threshold": val.get("contour_decode_threshold"),
            "max_centroid_error_bins": val.get("max_centroid_error_bins"),
            "max_length_error_frames": val.get("max_length_error_frames"),
        },
        "detector_parameters": {
            "type": det.get("type"),
            "freq_min_hz": det.get("freq_min_hz"),
            "freq_max_hz": det.get("freq_max_hz"),
            "min_contrast_db": det.get("min_contrast_db"),
            "max_gap_frames": det.get("max_gap_frames"),
            "background_normalization": det.get("background_normalization"),
        },
        "nfm_parameters": {
            "baseband_fs_hz": nfm.get("baseband_fs_hz"),
            "amplitude": nfm.get("amplitude"),
            "center_freq_definition": nfm.get("center_freq_definition"),
        },
        "software_versions": _software_versions(),
        "storage_note": "HDF5/parquet/versioning are engineering storage decisions, NOT paper claims.",
    }


def write_readme(version_dir, manifest):
    txt = (
        f"# Canonical preprocessing dataset {manifest['dataset_version']}\n\n"
        f"- Samples: {manifest['total_samples']}\n"
        f"- Source files: {manifest['source_file_count']} "
        f"(NOAA DCLDE 2022 Cruise 1705 Encounter 191, channel 0 only)\n"
        f"- Training input: `/images` in `dataset.h5`, shape "
        f"(N,128,128,1), float32, [0,1].\n"
        f"- Per-sample metadata: `metadata.parquet`.\n"
        f"- Provenance: `manifest.json` (config hash "
        f"{manifest['config_hash'][:12] if manifest['config_hash'] else None}, "
        f"H1 fingerprint "
        f"{(manifest['h1_fingerprint'][:12] if manifest['h1_fingerprint'] else None)}).\n"
        f"- QC: `qc.csv` (log-only; nothing was excluded by QC beyond the\n"
        f"  scientific L>128/margin rejects recorded upstream).\n"
        f"- `previews/` are debug visualizations, NOT training data.\n"
        f"- Storage format (HDF5/parquet) is an engineering decision, "
        f"not a paper claim.\n"
    )
    with open(os.path.join(version_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(txt)


def finalize_versioned_dataset(staging_dir, final_dir, version, cfg,
                               config_hash, h1_fingerprint, source_files):
    """Integrity-check staging, materialize parquet/csv/manifest, mark COMPLETE."""
    h5_path = os.path.join(staging_dir, "dataset.h5")
    if not os.path.exists(h5_path):
        raise ValueError("Missing dataset.h5 in staging")

    with h5py.File(h5_path, "r") as f:
        if "/images" not in f:
            raise ValueError("Missing /images in dataset.h5")
        d = f["/images"]
        n = int(d.shape[0])
        if tuple(d.shape[1:]) != (128, 128, 1):
            raise ValueError(f"Bad /images trailing shape {d.shape}")
        if str(d.dtype) != "float32":
            raise ValueError(f"Bad /images dtype {d.dtype}")
        # Chunked finite/range scan (never full-RAM).
        gmin, gmax = 1.0, 0.0
        step = 64
        for s in range(0, n, step):
            blk = d[s:s + step]
            if not np.all(np.isfinite(blk)):
                raise ValueError(f"Non-finite values in rows {s}:{s+step}")
            gmin = min(gmin, float(np.min(blk)) if blk.size else gmin)
            gmax = max(gmax, float(np.max(blk)) if blk.size else gmax)
        if n > 0 and (gmin < -1e-6 or gmax > 1.0 + 1e-6):
            raise ValueError(f"Values outside [0,1]: min={gmin} max={gmax}")

    meta_rows = _read_jsonl(os.path.join(staging_dir, "metadata_rows.jsonl"))
    qc_rows = _read_jsonl(os.path.join(staging_dir, "qc_rows.jsonl"))
    if len(meta_rows) != n:
        raise ValueError(f"metadata rows {len(meta_rows)} != N {n}")
    if len(qc_rows) != n:
        raise ValueError(f"qc rows {len(qc_rows)} != N {n}")
    if n:
        idxs = [r["dataset_index"] for r in meta_rows]
        if sorted(idxs) != list(range(n)):
            raise ValueError("dataset_index not unique/dense 0..N-1")
        ids = [r["image_id"] for r in meta_rows]
        if len(set(ids)) != n:
            raise ValueError("image_id not unique")

    meta_df = pd.DataFrame(meta_rows, columns=METADATA_COLUMNS)
    meta_df.to_parquet(os.path.join(staging_dir, "metadata.parquet"), index=False)
    qc_df = pd.DataFrame(qc_rows, columns=QC_COLUMNS)
    qc_df.to_csv(os.path.join(staging_dir, "qc.csv"), index=False)

    creation_ts = _utcnow()
    manifest = build_manifest(version, cfg, config_hash, h1_fingerprint,
                              source_files, n, creation_ts)
    with open(os.path.join(staging_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    write_readme(staging_dir, manifest)

    with open(os.path.join(staging_dir, "COMPLETE"), "w", encoding="utf-8") as f:
        f.write(f"COMPLETE {creation_ts} samples={n}\n")

    os.makedirs(os.path.dirname(os.path.abspath(final_dir)), exist_ok=True)
    if os.path.exists(final_dir):
        raise ValueError(f"Refusing to overwrite existing dataset dir: {final_dir}")
    os.rename(staging_dir, final_dir)
    return manifest


def dataset_info(version_dir):
    """Inspect an existing dataset version without preprocessing anything."""
    manifest_path = os.path.join(version_dir, "manifest.json")
    h5_path = os.path.join(version_dir, "dataset.h5")
    complete = os.path.exists(os.path.join(version_dir, "COMPLETE"))
    info = {"version_dir": version_dir, "complete": complete}
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            m = json.load(f)
        info.update({
            "dataset_version": m.get("dataset_version"),
            "total_samples": m.get("total_samples"),
            "source_file_count": m.get("source_file_count"),
            "image_shape": m.get("image_shape"),
            "image_dtype": m.get("image_dtype"),
            "normalization": m.get("normalization_method"),
            "config_hash": m.get("config_hash"),
            "h1_fingerprint": m.get("h1_fingerprint"),
            "creation_timestamp": m.get("creation_timestamp"),
        })
    else:
        # Incomplete staging dir: report live counts.
        n = None
        if os.path.exists(h5_path):
            with h5py.File(h5_path, "r") as f:
                if "/images" in f:
                    n = int(f["/images"].shape[0])
        info.update({"total_samples": n, "status": "INCOMPLETE"})
    if os.path.exists(h5_path):
        with h5py.File(h5_path, "r") as f:
            if "/images" in f:
                d = f["/images"]
                info["h5_shape"] = list(d.shape)
                info["h5_dtype"] = str(d.dtype)
                info["h5_chunks"] = list(d.chunks) if d.chunks else None
                info["h5_compression"] = d.compression
    return info
