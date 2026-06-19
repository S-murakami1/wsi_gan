from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
import torch


def list_h5_paths(directory: Path, glob_pattern: str) -> list[Path]:
    paths = sorted(directory.glob(glob_pattern))
    if not paths:
        raise FileNotFoundError(
            f"No HDF5 files matching {glob_pattern!r} under {directory.resolve()}"
        )
    return paths


class PatchIndexCatalog:
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
        file_idx = int(np.searchsorted(self._cumsum, flat_index, side="right"))
        start = int(self._cumsum[file_idx - 1]) if file_idx > 0 else 0
        row_idx = flat_index - start
        return file_idx, row_idx


class MultiH5PatchSampler:
    def __init__(
        self,
        catalog: PatchIndexCatalog,
        patches_dataset_key: str,
        device: torch.device,
        max_open_files: int = 64,
        seed: int | None = None,
    ):
        self._catalog = catalog
        self._key = patches_dataset_key
        self._device = device
        self._max_open = max(1, max_open_files)
        self._handles: OrderedDict[Path, h5py.File] = OrderedDict()
        self._rng = torch.Generator()
        if seed is not None:
            self._rng.manual_seed(seed)
        self.epoch = 0
        self._order: torch.Tensor | None = None
        self._pos = 0
        self._start_epoch()

    @property
    def position(self) -> int:
        return self._pos

    @property
    def total_patches(self) -> int:
        return self._catalog.total_patches

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

    def _load_flat_index(self, flat_index: int) -> torch.Tensor:
        file_idx, row_idx = self._catalog.file_and_row(flat_index)
        path = self._catalog.files[file_idx]
        ds = self._dataset(path)
        patch = np.asarray(ds[row_idx], dtype=np.uint8)
        t = torch.from_numpy(patch).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        t = t * 2.0 - 1.0
        return t.to(self._device)

    def _start_epoch(self) -> None:
        self.epoch += 1
        self._order = torch.randperm(self._catalog.total_patches, generator=self._rng)
        self._pos = 0

    def state_dict(self) -> dict[str, object]:
        if self._order is None:
            raise RuntimeError("sampler order is not initialized")
        return {
            "epoch": self.epoch,
            "pos": self._pos,
            "order": self._order.cpu(),
            "rng_state": self._rng.get_state(),
            "total_patches": self._catalog.total_patches,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        total = int(state["total_patches"])
        if total != self._catalog.total_patches:
            raise ValueError(
                f"sampler total_patches mismatch: checkpoint={total}, "
                f"current={self._catalog.total_patches}"
            )
        order = state["order"]
        if not isinstance(order, torch.Tensor):
            raise TypeError("sampler order must be a torch.Tensor")
        order = order.cpu()
        if order.numel() != total:
            raise ValueError(
                f"sampler order length mismatch: {order.numel()} != {total}"
            )
        pos = int(state["pos"])
        if pos < 0 or pos > order.numel():
            raise ValueError(f"sampler pos out of range: {pos}")
        self.epoch = int(state["epoch"])
        self._order = order
        self._pos = pos
        rng_state = state["rng_state"]
        if not isinstance(rng_state, torch.Tensor):
            rng_state = torch.as_tensor(rng_state, dtype=torch.uint8)
        else:
            rng_state = rng_state.to(dtype=torch.uint8)
        self._rng.set_state(rng_state.cpu())

    def _next_flat_indices(self, count: int) -> list[int]:
        if count < 1:
            raise ValueError("count must be >= 1")
        indices: list[int] = []
        while len(indices) < count:
            assert self._order is not None
            if self._pos >= len(self._order):
                self._start_epoch()
            take = min(count - len(indices), len(self._order) - self._pos)
            chunk = self._order[self._pos : self._pos + take].tolist()
            indices.extend(int(i) for i in chunk)
            self._pos += take
        return indices

    def sample_batch(self, batch_size: int) -> torch.Tensor:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        flats = self._next_flat_indices(batch_size)
        chunks = [self._load_flat_index(flat) for flat in flats]
        return torch.cat(chunks, dim=0)

    def close(self) -> None:
        for h5 in self._handles.values():
            h5.close()
        self._handles.clear()
