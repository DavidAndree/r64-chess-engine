"""Load encoded shards and feed shuffled batches to PyTorch.

Positions are stored compactly (bitboards, ~110 bytes each) and expanded to planes per batch,
so tens of millions of positions fit in memory.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np

from r64.data.encoding import planes_from_arrays
from r64.data.pgn import FIELDS


class PositionSet:
    """All positions from a set of shards, held in memory as numpy arrays."""

    def __init__(self, shard_paths: list[Path]):
        if not shard_paths:
            raise FileNotFoundError("no shards given")
        # Two passes (sizes, then fill) so memory peaks at the final size, not double.
        sizes = []
        for p in shard_paths:
            with np.load(p) as z:
                sizes.append(len(z["move"]))
        total = sum(sizes)
        self.arrays = {k: np.empty((total, *shape), dtype=dtype) for k, (dtype, shape) in FIELDS.items()}
        offset = 0
        for p, n in zip(shard_paths, sizes, strict=True):
            with np.load(p) as z:
                for k in FIELDS:
                    self.arrays[k][offset : offset + n] = z[k]
            offset += n

    @classmethod
    def from_dir(cls, directory: Path, split: str = "train") -> PositionSet:
        return cls(sorted(Path(directory).glob(f"{split}-*.npz")))

    def __len__(self) -> int:
        return len(self.arrays["move"])

    def game_ids(self) -> np.ndarray:
        """Game number of every position (positions of a game are stored in order: ply restarts
        at each new game)."""
        ply = self.arrays["ply"]
        return np.cumsum(np.r_[True, ply[1:] <= ply[:-1]]) - 1

    def select(self, mask: np.ndarray) -> np.ndarray:
        """Indices of positions passing a boolean mask, e.g. `ps.arrays["ply"] >= 10`."""
        return np.nonzero(mask)[0]

    def batch(self, idx: np.ndarray) -> dict[str, np.ndarray]:
        a = self.arrays
        return {
            "planes": planes_from_arrays(a["pieces"][idx], a["castling"][idx], a["ep"][idx]),
            "elo": a["elo"][idx].astype(np.float32),
            "opp_elo": a["opp_elo"][idx].astype(np.float32),
            "clock": a["clock"][idx].astype(np.float32),
            "base_time": a["base_time"][idx].astype(np.float32),
            "move": a["move"][idx].astype(np.int64),
            "result": a["result"][idx].astype(np.float32),
        }

    def batches(
        self,
        batch_size: int,
        indices: np.ndarray | None = None,
        shuffle: bool = True,
        seed: int | None = None,
        device=None,
    ) -> Iterator[dict]:
        """One pass over the data. With `device`, yields torch tensors on that device."""
        idx = np.arange(len(self)) if indices is None else indices
        if shuffle:
            idx = np.random.default_rng(seed).permutation(idx)
        for start in range(0, len(idx), batch_size):
            b = self.batch(np.sort(idx[start : start + batch_size]))
            if device is not None:
                import torch

                b = {k: torch.from_numpy(v).to(device, non_blocking=True) for k, v in b.items()}
            yield b
