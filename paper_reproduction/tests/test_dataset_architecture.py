"""
test_dataset_architecture.py -- Synthetic tests for the versioned canonical
dataset architecture (STORAGE ONLY). No FLAC access. No real preprocessing.

Covers: HDF5 append/lazy-read, schema, metadata/manifest/COMPLETE,
PyTorch loader batches, version overwrite refusal, incomplete rejection,
training reference linkage.
"""

import os
import sys
import json
import shutil
import tempfile
import unittest

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from dataset_store import (
    VersionedDatasetWriter,
    finalize_versioned_dataset,
    dataset_info,
    METADATA_COLUMNS,
)
from whistle_dataset import (
    WhistleHDF5Dataset,
    make_dataloader,
    write_preprocessing_reference,
)


def _mini_cfg(version="preprocessing_v001"):
    return {
        "version": "2.0.0-final",
        "input": {"expected_fs": 500000},
        "resample": {"fs_out": 192000},
        "channel_handling": {"channel_index": 0},
        "detector": {
            "type": "classical_ridge_tracker_standin",
            "freq_min_hz": 2000.0,
            "freq_max_hz": 20000.0,
            "min_contrast_db": 4.0,
            "max_gap_frames": 2,
            "background_normalization": "per_bin_median",
        },
        "nfm": {"baseband_fs_hz": 12000.0, "amplitude": 1.0,
                "center_freq_definition": "midrange"},
        "image": {"normalization": "calibrated_two_sided_window_sum"},
        "validation": {"contour_decode_threshold": 0.05,
                       "max_centroid_error_bins": 1.0,
                       "max_length_error_frames": 3},
        "dataset": {"version": version},
    }


def _synth_image(rng, bright=True):
    img = rng.uniform(0.0, 0.04, size=(128, 128)).astype(np.float32)
    if bright:
        r = rng.integers(40, 88)
        img[r - 1:r + 2, 30:98] = rng.uniform(0.6, 1.0, size=(3, 68)).astype(np.float32)
    return np.clip(img, 0.0, 1.0)


def _rows(i, src="synth_file.flac"):
    meta = {
        "source_file": src, "source_channel": 0, "contour_id": f"c_{i:04d}",
        "duration_frames": 68, "duration_seconds": 0.3626,
        "fmin_hz": 5000.0, "fmax_hz": 8000.0, "center_frequency_hz": 6500.0,
        "bandwidth_hz": 3000.0, "start_frequency_hz": 5100.0,
        "end_frequency_hz": 7900.0, "image_height": 128, "image_width": 128,
        "dtype": "float32", "normalization": "calibrated_two_sided_window_sum",
        "preprocessing_version": "preprocessing_v001",
        "config_hash": "deadbeef", "h1_fingerprint": "feedface",
        "qc_status": "pass",
    }
    qc = {
        "contour_id": f"c_{i:04d}", "source_file": src, "channel": 0,
        "status": "pass", "original_length": 68, "recovered_length": 70,
        "length_error_frames": 2, "max_centroid_error_bins": 0.1,
        "mean_centroid_error_bins": 0.02, "peak_intensity": 0.99,
    }
    return meta, qc


class TestDatasetArchitecture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ds_arch_")
        self.rng = np.random.default_rng(0)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_version(self, version, n_total, cfg=None):
        cfg = cfg or _mini_cfg(version)
        staging = os.path.join(self.tmp, version + ".tmp")
        final = os.path.join(self.tmp, version)
        w = VersionedDatasetWriter(
            staging, version, cfg, "deadbeef", "feedface",
            ["a.flac", "b.flac"], chunk_samples=8,
            compression="gzip", compression_level=4, preview_count=4)
        # 2. append in multiple batches (memory-efficient path)
        for i in range(n_total):
            w.append_sample(_synth_image(self.rng), *_rows(i), "a.flac")
        w.mark_file_complete("a.flac", ["a.flac", "b.flac"])
        manifest = finalize_versioned_dataset(
            staging, final, version, cfg, "deadbeef", "feedface",
            ["a.flac", "b.flac"])
        return final, manifest

    def test_01_09_canonical_schema_and_manifest(self):
        final, manifest = self._make_version("preprocessing_v001", 20)
        import h5py
        with h5py.File(os.path.join(final, "dataset.h5"), "r") as f:
            d = f["/images"]
            self.assertEqual(d.shape, (20, 128, 128, 1))  # 3,4 exact schema
            self.assertEqual(str(d.dtype), "float32")
            self.assertIsNotNone(d.chunks)  # chunked
            self.assertEqual(d.compression, "gzip")
            blk = d[0:20]  # 5. batch access without full-RAM requirement
            self.assertTrue(np.all(np.isfinite(blk)))
            self.assertGreaterEqual(float(np.min(blk)), -1e-6)
            self.assertLessEqual(float(np.max(blk)), 1.0 + 1e-6)
            one = d[7]  # indexed access
            self.assertEqual(one.shape, (128, 128, 1))
        import pandas as pd
        meta = pd.read_parquet(os.path.join(final, "metadata.parquet"))
        self.assertEqual(len(meta), 20)  # 6. row count
        self.assertEqual(list(meta.columns), METADATA_COLUMNS)
        self.assertEqual(meta["dataset_index"].tolist(), list(range(20)))
        self.assertEqual(len(set(meta["image_id"])), 20)
        # 6. manifest + 7. COMPLETE
        self.assertEqual(manifest["total_samples"], 20)
        self.assertTrue(os.path.exists(os.path.join(final, "COMPLETE")))
        self.assertTrue(os.path.exists(os.path.join(final, "README.md")))
        self.assertTrue(os.path.exists(os.path.join(final, "qc.csv")))
        self.assertIn("config_hash", manifest)
        self.assertIn("h1_fingerprint", manifest)

    def test_10_torch_loader_batches(self):
        import torch
        final, _ = self._make_version("preprocessing_v001", 20)
        ds = WhistleHDF5Dataset(final)
        self.assertEqual(len(ds), 20)
        x, idx = ds[0]
        self.assertIsInstance(x, torch.Tensor)
        self.assertEqual(tuple(x.shape), (1, 128, 128))  # 9. (C,H,W)
        self.assertEqual(x.dtype, torch.float32)
        self.assertGreaterEqual(float(x.min()), -1e-6)
        self.assertLessEqual(float(x.max()), 1.0 + 1e-6)
        dl = make_dataloader(final, batch_size=8, shuffle=False, num_workers=0)
        batches = list(dl)
        self.assertEqual(len(batches), 3)  # 10. multiple batches 8/8/4
        self.assertEqual(tuple(batches[0][0].shape), (8, 1, 128, 128))
        dl2 = make_dataloader(final, batch_size=7, shuffle=True, num_workers=2)
        n = sum(b[0].shape[0] for b in dl2)
        self.assertEqual(n, 20)

    def test_11_version_no_overwrite(self):
        final, _ = self._make_version("preprocessing_v001", 5)
        cfg = _mini_cfg("preprocessing_v002")
        # A second version finalizes to its own directory; v001 untouched.
        final2, _ = self._make_version("preprocessing_v002", 3)
        self.assertTrue(os.path.exists(os.path.join(final, "COMPLETE")))
        self.assertTrue(os.path.exists(os.path.join(final2, "COMPLETE")))
        import h5py
        with h5py.File(os.path.join(final, "dataset.h5"), "r") as f:
            self.assertEqual(int(f["/images"].shape[0]), 5)
        # Finalizing onto an existing final dir must refuse.
        with self.assertRaises(ValueError):
            finalize_versioned_dataset(
                os.path.join(self.tmp, "preprocessing_v002.tmp"),
                final2, "preprocessing_v002", cfg,
                "deadbeef", "feedface", ["a.flac"])

    def test_12_incomplete_not_valid(self):
        cfg = _mini_cfg("preprocessing_v001")
        staging = os.path.join(self.tmp, "preprocessing_v001.tmp")
        w = VersionedDatasetWriter(staging, "preprocessing_v001", cfg,
                                   "deadbeef", "feedface", ["a.flac"])
        w.append_sample(_synth_image(self.rng), *_rows(0), "a.flac")
        info = dataset_info(staging)
        self.assertFalse(info["complete"])
        with self.assertRaises(ValueError):
            WhistleHDF5Dataset(staging)  # 12. loader refuses incomplete

    def test_13_training_reference(self):
        final, manifest = self._make_version("preprocessing_v001", 6)
        exp_dir = os.path.join(self.tmp, "experiments", "vae_v001")
        ref_path = write_preprocessing_reference(exp_dir, final)
        with open(ref_path, "r", encoding="utf-8") as f:
            ref = json.load(f)
        self.assertEqual(ref["dataset_version"], "preprocessing_v001")
        self.assertEqual(ref["dataset_path"], os.path.abspath(final))
        self.assertEqual(ref["total_samples"], 6)
        self.assertIn("manifest_sha256", ref)
        self.assertEqual(ref["manifest_config_hash"], manifest["config_hash"])

    def test_14_dataset_info_reuse(self):
        final, manifest = self._make_version("preprocessing_v001", 4)
        info = dataset_info(final)
        self.assertTrue(info["complete"])
        for k in ("dataset_version", "total_samples", "source_file_count",
                  "image_shape", "image_dtype", "normalization",
                  "config_hash", "h1_fingerprint", "creation_timestamp"):
            self.assertIn(k, info)
        self.assertEqual(info["total_samples"], 4)


if __name__ == "__main__":
    unittest.main()
