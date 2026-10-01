"""CEBaB (``cebab``): restaurant review -> 5 aspect sentiments -> overall label.

Each of the 5 concepts (food, service, noise, ambiance, review majority) takes
a value in Positive=0, Negative=1, Neutral=2, unknown=3; the 1-5 star review
rating is bucketed into the same values (4-5, 3, 1-2, missing). Items are
``(tokenized_text, concept_indices[5])``; the label is derived by the oracle.
Data (CEBaB v1.1) is downloaded by ``make cebab`` into ``fonesy/data/cebab/``.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from fonesy.datasets.tasks.base import Task

_MAX_LENGTH = 128
_TOKENIZER_NAME = "bert-base-uncased"

_ASPECTS = ("food", "service", "noise", "ambiance")

# Oracle value order: Positive=0, Negative=1, Neutral=2, unknown=3.
_ASPECT_TO_IDX = {
    "Positive": 0,
    "Negative": 1,
    "Neutral": 2,
    "unknown": 3,
    "no majority": 3,
    "": 3,
}


def _review_to_idx(review_majority: str) -> int:
    """Bucket a 1-5 star ``review_majority`` string into the 4-valued concept."""
    if review_majority in ("4", "5"):
        return 0  # Positive
    if review_majority == "3":
        return 2  # Neutral
    if review_majority in ("1", "2"):
        return 1  # Negative
    return 3  # unknown / "no majority" / ""


def entry_to_concepts(entry: dict) -> list[int]:
    """CeBaB JSON entry -> ``[food, service, noise, ambiance, review]`` concept indices."""
    concepts = [
        _ASPECT_TO_IDX.get((entry.get(f"{a}_aspect_majority") or "").strip(), 3)
        for a in _ASPECTS
    ]
    concepts.append(_review_to_idx((entry.get("review_majority") or "").strip()))
    return concepts


class _CeBaBSplit(Dataset):
    """Pre-tokenised split yielding ``(stacked_ids_mask[L, 2], concepts[5])``."""

    def __init__(self, entries: list[dict], tokenizer) -> None:
        texts = [e["description"] for e in entries]
        self._concepts = torch.tensor(
            [entry_to_concepts(e) for e in entries], dtype=torch.long
        )
        tok = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=_MAX_LENGTH,
            return_tensors="pt",
        )
        # (N, L, 2): [..., 0] input_ids, [..., 1] attention_mask
        self._tokens = torch.stack([tok["input_ids"], tok["attention_mask"]], dim=-1)

    def __len__(self) -> int:
        return self._concepts.shape[0]

    def __getitem__(self, idx: int):
        return self._tokens[idx], self._concepts[idx]


class CeBaBTask(Task):
    """CeBaB review-sentiment task (text -> 5 concepts -> label)."""

    n_classes: int = 5   # sentiment labels; unused by SPL directly
    in_channels: int = 1  # unused
    image_size: int = 1   # unused
    feature_dim: int = 256  # BertBackbone projection width (see fonesy/networks/bert.py)

    _SPLIT_FILES = {
        "train": "train_observational.json",
        "val": "dev.json",
        "test": "test.json",
    }

    def __init__(self, root: str) -> None:
        import json

        data_dir = Path(root) / "cebab"
        missing = [f for f in self._SPLIT_FILES.values() if not (data_dir / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"CeBaB data not found at '{data_dir}' (missing: {', '.join(missing)}).\n"
                "Run 'make cebab'."
            )

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER_NAME)
        self._splits = {}
        for split, fname in self._SPLIT_FILES.items():
            entries = json.loads((data_dir / fname).read_text())
            self._splits[split] = _CeBaBSplit(entries, tokenizer)

    def train_loader(self, batch_size: int) -> DataLoader:
        return DataLoader(self._splits["train"], batch_size=batch_size, shuffle=True)

    def val_loader(self, batch_size: int) -> DataLoader:
        return DataLoader(self._splits["val"], batch_size=batch_size, shuffle=False)

    def test_loader(self, batch_size: int) -> DataLoader:
        return DataLoader(self._splits["test"], batch_size=batch_size, shuffle=False)
