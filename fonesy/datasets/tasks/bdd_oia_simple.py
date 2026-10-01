"""BDD-OIA restricted to the move-forward/stop concepts (``bdd-oia-2``).

The 9 concepts of the ``bdd-oia-2`` oracle are the first 9 (oracle-ordered)
concepts of :class:`BddOiaTask`.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from fonesy.datasets.tasks.bdd_oia import BddOiaTask

N_CONCEPTS_SIMPLE = 9


class _BddOiaSimpleSplit(Dataset):
    """Wraps a full-BDD-OIA split, slicing concepts to the first 9 columns."""

    def __init__(self, inner) -> None:
        self._inner = inner

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, idx: int):
        features, concepts = self._inner[idx]
        return features, concepts[:N_CONCEPTS_SIMPLE]


class BddOiaSimpleTask(BddOiaTask):
    """BDD-OIA-Simple: precomputed 2048-d Faster-RCNN features, 9 concepts.

    Reuses ``BddOiaTask``'s data loading (same on-disk layout, same
    ``fonesy/data/bdd-oia/bdd2048/`` path) and slices the concept vector.
    """

    def _split_dataset(self, split: str) -> _BddOiaSimpleSplit:
        return _BddOiaSimpleSplit(super()._split_dataset(split))

    def train_loader(self, batch_size: int) -> DataLoader:
        return DataLoader(
            self._split_dataset("train"), batch_size=batch_size, shuffle=True,
            num_workers=2, pin_memory=torch.cuda.is_available(),
        )

    def val_loader(self, batch_size: int) -> DataLoader:
        return DataLoader(
            self._split_dataset("val"), batch_size=batch_size, shuffle=False,
            num_workers=2, pin_memory=torch.cuda.is_available(),
        )

    def test_loader(self, batch_size: int) -> DataLoader:
        return DataLoader(
            self._split_dataset("test"), batch_size=batch_size, shuffle=False,
            num_workers=2, pin_memory=torch.cuda.is_available(),
        )
