"""BDD-OIA (``bdd-oia``): precomputed 2048-d Faster-RCNN features -> 21 concepts.

Uses the features released with rsbench, extracted by ``make bdd-oia`` to::

    fonesy/data/bdd-oia/bdd2048/{train,val,test}_BDD_OIA.pkl
    fonesy/data/bdd-oia/bdd2048/{train,val,test}/{inputs,concepts}/<id>.pt

Only the concepts are read (actions are derived from them by the oracle);
their columns are reordered to the oracle's ``VARIABLE_NAMES`` (see
``_CONCEPT_PERMUTATION``).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from fonesy.datasets.tasks.base import Task
from fonesy.utils.oracles import load_oracle

_ORACLE_VARIABLE_NAMES = load_oracle("bdd-oia").VARIABLE_NAMES

N_CONCEPTS = 21

# rsbench-code's CONCEPTS_ORDER (rsseval/rss/datasets/utils/sddoia_creation.py),
# shared by boia.py/boia_creation.py — concept name -> feature-tensor column index.
_RSBENCH_CONCEPTS_ORDER = {
    "red_light": 3, "green_light": 0, "car": 5, "person": 6, "rider": 7,
    "other_obstacle": 8, "follow": 1, "stop_sign": 4,
    "left_lane": 18, "left_green_light": 19, "left_follow": 20,
    "no_left_lane": 9, "left_obstacle": 10, "left_solid_line": 11,
    "right_lane": 12, "right_green_light": 13, "right_follow": 14,
    "no_right_lane": 15, "right_obstacle": 16, "right_solid_line": 17,
    "clear": 2,
}
# rsbench name -> CNF oracle name, for the two concepts named differently.
_RSBENCH_TO_ORACLE_NAME = {"clear": "road_clear", "stop_sign": "traffic_sign"}


def _build_concept_permutation() -> list[int]:
    """rsbench column order -> oracle ``VARIABLE_NAMES`` order, index list."""
    oracle_concept_names = _ORACLE_VARIABLE_NAMES[:N_CONCEPTS]
    oracle_to_rsbench = {v: k for k, v in _RSBENCH_TO_ORACLE_NAME.items()}
    perm = []
    for oracle_name in oracle_concept_names:
        rsbench_name = oracle_to_rsbench.get(oracle_name, oracle_name)
        perm.append(_RSBENCH_CONCEPTS_ORDER[rsbench_name])
    assert sorted(perm) == list(range(N_CONCEPTS)), "concept permutation is not a bijection"
    return perm


_CONCEPT_PERMUTATION = _build_concept_permutation()


class _BddOiaSplit(Dataset):
    """One split (train/val/test) of precomputed BDD-OIA features + concepts."""

    def __init__(self, split_dir: Path, pkl_path: Path, permutation: list[int]) -> None:
        with open(pkl_path, "rb") as f:
            entries = pickle.load(f)
        self._ids = [Path(e["img_path"]).stem for e in entries]
        self._inputs_dir = split_dir / "inputs"
        self._concepts_dir = split_dir / "concepts"
        self._permutation = torch.tensor(permutation, dtype=torch.long)

    def __len__(self) -> int:
        return len(self._ids)

    def __getitem__(self, idx: int):
        sample_id = self._ids[idx]
        features = torch.load(
            self._inputs_dir / f"{sample_id}.pt", weights_only=True
        ).squeeze(0)
        concepts = torch.load(
            self._concepts_dir / f"{sample_id}.pt", weights_only=True
        ).squeeze(0)
        concepts = concepts[self._permutation]
        return features.float(), concepts.float()


class BddOiaTask(Task):
    """BDD-OIA: precomputed 2048-d Faster-RCNN features, 21 concepts.

    No CNN backbone — loaders yield ``(features, concept_bits)`` 2-tuples fed
    directly to the SPL head. ``n_classes``/``in_channels``/``image_size`` are
    unused placeholders (satisfying the ``Task`` contract); use
    ``feature_dim`` to size the head's input layer.
    """

    n_classes: int = 2       # unused — no CNN/classifier for this task
    in_channels: int = 1     # unused
    image_size: int = 1      # unused
    feature_dim: int = 2048

    def __init__(self, root: str) -> None:
        dataset_root = Path(root) / "bdd-oia" / "bdd2048"
        if not dataset_root.exists():
            raise FileNotFoundError(
                f"BDD-OIA precomputed features not found at '{dataset_root}'.\n"
                "Run 'make bdd-oia BDD_OIA_ZIP=<path or URL of bdd_2048.zip>'."
            )
        self._root = dataset_root

    def _split_dataset(self, split: str) -> _BddOiaSplit:
        split_dir = self._root / split
        pkl_path = self._root / f"{split}_BDD_OIA.pkl"
        return _BddOiaSplit(split_dir, pkl_path, _CONCEPT_PERMUTATION)

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
