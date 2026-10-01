"""Warcraft shortest path (``warcraft``): 4x4 terrain map -> path edges.

Each cell gets one of 5 terrain classes (cost 4^k) uniformly at random and is
rendered with a real Warcraft II tile (from ``fonesy/data/warcraft/train.npz``,
``make warcraft``; flat colors are used if the file is missing). The label is
the 24-edge encoding of the minimum-cost simple path from (0, 0) to (3, 3).
Items are ``(image[3, 64, 64], path_bits[24])``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from fonesy.utils.oracles import load_oracle

_oracle = load_oracle("warcraft")
N_VARS = _oracle.N_VARS

from fonesy.datasets.tasks.base import Task

_IMG = 64
_TILE = 16  # rendered px per grid cell (8px sprite upscaled x2)
_GRID = 4

_N_TRAIN = 8_000
_N_VAL = 1_000
_N_TEST = 1_000

_WARCRAFT_NPZ = Path(__file__).resolve().parents[2] / "data" / "warcraft" / "train.npz"
_TILE_BANK_MAPS = 500

# 5 SP(12) terrain-cost levels (see the dataset's `costs` array), ascending —
# used only to bucket real sprites by terrain type.
_COST_LEVELS = (0.8, 1.2, 5.301, 7.699, 9.203)
# Traversal cost per level for the shortest-path label. Steep (4**k) so that
# routing around one expensive tile beats a several-cell detour — at k=4 the
# raw SP(12) values are too flat to ever prefer a longer path over the
# 6-edge staircase, which collapses every label onto a shortest path.
_TRAVERSAL_COST = np.array([1.0, 4.0, 16.0, 64.0, 256.0], dtype=np.float32)
# Flat fallback colours, one per level (dark green / brown / green / blue / grey).
_FALLBACK_RGB = np.array(
    [[48, 128, 32], [128, 96, 24], [24, 110, 24], [24, 70, 150], [120, 120, 120]],
    dtype=np.float32,
) / 255.0


def _enumerate_paths() -> list[tuple[list[tuple[int, int]], np.ndarray]]:
    """Every simple ``(0,0)->(3,3)`` path as ``(vertex_list, edge_bits[24])``."""
    out: list[tuple[list[tuple[int, int]], np.ndarray]] = []

    def dfs(cur, visited, path):
        if cur == (_GRID - 1, _GRID - 1):
            bits = np.zeros(N_VARS, dtype=np.float32)
            for (r0, c0), (r1, c1) in zip(path, path[1:]):
                if r0 == r1:
                    var = _oracle._hvar(r0, min(c0, c1))
                else:
                    var = _oracle._vvar(min(r0, r1), c0)
                bits[var - 1] = 1.0
            assert _oracle.check([int(b) for b in bits])
            out.append((list(path), bits))
            return
        r, c = cur
        for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if 0 <= nr < _GRID and 0 <= nc < _GRID and (nr, nc) not in visited:
                visited.add((nr, nc))
                path.append((nr, nc))
                dfs((nr, nc), visited, path)
                path.pop()
                visited.remove((nr, nc))

    dfs((0, 0), {(0, 0)}, [(0, 0)])
    return out


_PATHS = _enumerate_paths()
_PATH_VERTICES = [v for v, _ in _PATHS]
_PATH_BITS = np.stack([b for _, b in _PATHS])  # (184, 24)


def _shortest_path_bits(cell_cost: np.ndarray) -> np.ndarray:
    """24-bit indicator of the min vertex-cost simple path through ``cell_cost`` (4x4)."""
    best_i, best_c = 0, np.inf
    for i, verts in enumerate(_PATH_VERTICES):
        c = sum(cell_cost[r, k] for r, k in verts)
        if c < best_c:
            best_c, best_i = c, i
    return _PATH_BITS[best_i]


_TILE_BANK: dict[int, np.ndarray] | None = None  # {cost_level_idx: (K, _TILE, _TILE, 3) float}


def _load_tile_bank() -> dict[int, np.ndarray] | None:
    global _TILE_BANK
    if _TILE_BANK is not None:
        return _TILE_BANK
    if not _WARCRAFT_NPZ.exists():
        return None
    d = np.load(_WARCRAFT_NPZ)
    maps = d["maps"][:_TILE_BANK_MAPS].astype(np.float32) / 255.0  # (N,96,96,3)
    costs = d["costs"][:_TILE_BANK_MAPS].astype(np.float32)  # (N,12,12)
    n = maps.shape[0]
    tiles = (
        maps.reshape(n, 12, 8, 12, 8, 3).transpose(0, 1, 3, 2, 4, 5).reshape(-1, 8, 8, 3)
    )
    up = _TILE // 8
    tiles = np.repeat(np.repeat(tiles, up, axis=1), up, axis=2)
    flat_cost = costs.reshape(-1)
    bank: dict[int, np.ndarray] = {}
    for idx, level in enumerate(_COST_LEVELS):
        sel = np.abs(flat_cost - level) < 0.05
        bank[idx] = np.ascontiguousarray(tiles[sel][:4000])
    _TILE_BANK = bank
    return _TILE_BANK


class _WarcraftSplit(Dataset):
    """``n`` deterministic ``(image[3,64,64], shortest_path_bits[24])`` samples."""

    def __init__(self, n: int, seed: int) -> None:
        self._n = n
        self._seed = seed
        self._bank = _load_tile_bank()

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int):
        rng = np.random.RandomState(self._seed * 1_000_003 + idx)
        level_idx = rng.randint(0, len(_COST_LEVELS), size=(_GRID, _GRID))
        cell_cost = _TRAVERSAL_COST[level_idx]

        canvas = np.zeros((_IMG, _IMG, 3), dtype=np.float32)
        for r in range(_GRID):
            for c in range(_GRID):
                li = int(level_idx[r, c])
                if self._bank is not None and len(self._bank[li]) > 0:
                    pool = self._bank[li]
                    tile = pool[rng.randint(0, pool.shape[0])]
                else:
                    tile = np.broadcast_to(_FALLBACK_RGB[li], (_TILE, _TILE, 3))
                canvas[r * _TILE : (r + 1) * _TILE, c * _TILE : (c + 1) * _TILE] = tile

        path_bits = _shortest_path_bits(cell_cost)
        img = np.ascontiguousarray(canvas.transpose(2, 0, 1))
        return torch.from_numpy(img), torch.from_numpy(path_bits.copy())


class WarcraftPathTask(Task):
    """4x4 Warcraft shortest-path over real terrain tiles (24 edge variables)."""

    n_classes: int = 2   # unused — SPL scores the 24-bit assignment
    in_channels: int = 3
    image_size: int = _IMG

    def __init__(self, root: str) -> None:  # noqa: ARG002 - data path is fixed
        self._train = _WarcraftSplit(_N_TRAIN, seed=1)
        self._val = _WarcraftSplit(_N_VAL, seed=2)
        self._test = _WarcraftSplit(_N_TEST, seed=3)
        self.uses_real_tiles = self._train._bank is not None

    def train_loader(self, batch_size: int) -> DataLoader:
        return DataLoader(self._train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())

    def val_loader(self, batch_size: int) -> DataLoader:
        return DataLoader(self._val, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())

    def test_loader(self, batch_size: int) -> DataLoader:
        return DataLoader(self._test, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())
