"""
whistle_dataset.py -- PyTorch loader for the canonical HDF5 whistle dataset.

Reads ONLY the stored [0,1] float32 representation. No renormalization,
no FLAC/STFT/NFM recomputation during training. HDF5 versioning/provenance
are engineering decisions, NOT paper claims.
"""

import os
import json

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class WhistleHDF5Dataset(Dataset):
    """Lazy HDF5 whistle dataset. Returns (1,128,128) float32 in [0,1]."""

    def __init__(self, version_dir, transform=None):
        self.version_dir = os.path.abspath(version_dir)
        self.h5_path = os.path.join(self.version_dir, "dataset.h5")
        if not os.path.exists(os.path.join(self.version_dir, "COMPLETE")):
            raise ValueError(
                f"Dataset {self.version_dir} is not COMPLETE; "
                "refusing to train on a partial dataset."
            )
        with h5py.File(self.h5_path, "r") as f:
            self._len = int(f["/images"].shape[0])
        self.transform = transform
        self._h5 = None  # worker-local handle, never pickled

    def __getstate__(self):
        st = self.__dict__.copy()
        st["_h5"] = None
        return st

    def _handle(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        if idx < 0:
            idx = self._len + idx
        if not 0 <= idx < self._len:
            raise IndexError(idx)
        arr = self._handle()["/images"][idx]  # (128,128,1), no full-RAM load
        t = torch.from_numpy(np.asarray(arr, dtype=np.float32)).permute(2, 0, 1)
        if self.transform is not None:
            t = self.transform(t)
        return t, int(idx)


def make_dataloader(version_dir, batch_size=32, shuffle=True, num_workers=0,
                    **kwargs):
    ds = WhistleHDF5Dataset(version_dir)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, **kwargs)


def write_preprocessing_reference(experiment_dir, version_dir):
    """Record which canonical dataset a training experiment used."""
    version_dir = os.path.abspath(version_dir)
    with open(os.path.join(version_dir, "manifest.json"), "r", encoding="utf-8") as f:
        m = json.load(f)
    ref = {
        "dataset_version": m.get("dataset_version"),
        "dataset_path": version_dir,
        "total_samples": m.get("total_samples"),
        "manifest_config_hash": m.get("config_hash"),
        "manifest_h1_fingerprint": m.get("h1_fingerprint"),
        "manifest_creation": m.get("creation_timestamp"),
    }
    # Content hash of manifest for tamper evidence.
    import hashlib
    h = hashlib.sha256()
    with open(os.path.join(version_dir, "manifest.json"), "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    ref["manifest_sha256"] = h.hexdigest()
    os.makedirs(experiment_dir, exist_ok=True)
    out = os.path.join(experiment_dir, "preprocessing_reference.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(ref, f, indent=2)
    return out
