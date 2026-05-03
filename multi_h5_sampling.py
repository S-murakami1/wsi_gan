"""Random patch sampling from many HDF5 slide files in one directory.

Each training step picks one flat index uniformly from all patches across files,
then maps it to (file, row). Open HDF5 handles are cached with a small LRU cap
to avoid reopening every step while bounding file-descriptor use.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
import torch


def list_h5_paths(directory: Path, glob_pattern: str) -> list[Path]:
    """Sorted paths under ``directory`` matching ``glob_pattern`` (e.g. ``*.h5``)."""
    paths = sorted(directory.glob(glob_pattern))
    if not paths:
        raise FileNotFoundError(
            f"No HDF5 files matching {glob_pattern!r} under {directory.resolve()}"
        )
    return paths


class PatchIndexCatalog:
    """Per-file patch counts and mapping from one global index to (file_idx, row_idx)."""

    def __init__(self, h5_paths: list[Path], patches_dataset_key: str):
        self.files = list(h5_paths)
        counts: list[int] = []
        for path in self.files:
            with h5py.File(path, "r") as f:
                n = int(f[patches_dataset_key].shape[0])
            if n <= 0:
                raise ValueError(f"Empty patch dataset in {path}")
            counts.append(n)
        self._counts = np.asarray(counts, dtype=np.int64)
        self._cumsum = np.cumsum(self._counts)
        self.total_patches = int(self._cumsum[-1])

    def file_and_row(self, flat_index: int) -> tuple[int, int]:
        """Map ``flat_index`` in ``[0, total_patches)`` to file index and row index."""
        file_idx = int(np.searchsorted(self._cumsum, flat_index, side="right"))
        start = int(self._cumsum[file_idx - 1]) if file_idx > 0 else 0
        row_idx = flat_index - start
        return file_idx, row_idx


class RandomMultiH5PatchSampler:
    """Draw one random patch per call, uniformly over all patches in the catalog."""

    def __init__(
        self,
        catalog: PatchIndexCatalog,
        patches_dataset_key: str,
        device: torch.device,
        max_open_files: int = 64,
    ):
        self._catalog = catalog
        self._key = patches_dataset_key
        self._device = device
        self._max_open = max(1, max_open_files)
        self._handles: OrderedDict[Path, h5py.File] = OrderedDict()

    def _dataset(self, path: Path) -> h5py.Dataset:
        if path in self._handles:
            self._handles.move_to_end(path)
            return self._handles[path][self._key]
        while len(self._handles) >= self._max_open:
            _oldest_path, old_file = self._handles.popitem(last=False)
            old_file.close()
        h5 = h5py.File(path, "r")
        self._handles[path] = h5
        return h5[self._key]

    def sample(self) -> torch.Tensor:
        flat = int(torch.randint(0, self._catalog.total_patches, (1,)).item())
        file_idx, row_idx = self._catalog.file_and_row(flat)
        path = self._catalog.files[file_idx]
        ds = self._dataset(path)
        patch = np.asarray(ds[row_idx], dtype=np.uint8)
        t = torch.from_numpy(patch).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        t = t * 2.0 - 1.0
        return t.to(self._device)

    def close(self) -> None:
        for h5 in self._handles.values():
            h5.close()
        self._handles.clear()
