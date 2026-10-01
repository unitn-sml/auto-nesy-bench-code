"""ChestX-Rays (``chx``): chest X-ray -> 4 findings -> 5-way severity code.

Findings (fracture, pneumothorax, airspace opacity, nodule/mass) come from the
radiologist-adjudicated labels of ChestX-ray14, majority-voted across readers.
Items are ``(image[3, 224, 224], findings[4])``. Data: ``make chx`` (requires
an authenticated ``gcloud`` for the label CSVs).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from fonesy.datasets.tasks.base import Task

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# CSV column -> oracle finding order (chx.VARIABLE_NAMES[:4]).
_FINDING_COLS = ["Fracture", "Pneumothorax", "Airspace opacity", "Nodule/mass"]

_LABEL_CSVS = [
    "four_findings_expert_labels_individual_readers.csv",
    "four_findings_expert_labels_test_labels.csv",
    "four_findings_expert_labels_validation_labels.csv",
]

_TEST_SPLIT = 0.2
_VAL_SPLIT = 0.1


def _find_image_dir(data_dir: Path) -> Path | None:
    """Locate the annotated-image directory under either the ConformalNeSy
    layout (``images_nih/images/``) or the flat ``images/`` layout."""
    for candidate in (data_dir / "images_nih" / "images", data_dir / "images"):
        if candidate.is_dir():
            return candidate
    return None


def _majority_vote_findings(data_dir: Path) -> "pd.DataFrame":  # noqa: F821
    """Aggregate the four-findings CSVs to one 0/1 row per image.

    Normalises the two on-disk schemas: ``four_findings_expert_labels_individual_readers.csv``
    keys images by ``Image ID`` and uses ``Nodule/mass``; the adjudicated
    ``*_test_labels.csv`` / ``*_validation_labels.csv`` key by ``Image Index``
    and use ``Nodule or mass``. All rows are pooled and majority-voted per
    image (matching ConformalNeSy's ``CHXLoader``).
    """
    import pandas as pd

    frames = []
    for name in _LABEL_CSVS:
        path = data_dir / name
        if path.exists():
            df = pd.read_csv(path)
            df = df.rename(
                columns={"Image Index": "Image ID", "Nodule or mass": "Nodule/mass"}
            )
            frames.append(df[["Image ID", *_FINDING_COLS]])
    if not frames:
        raise FileNotFoundError(
            f"No four-findings label CSVs found in '{data_dir}'. Expected any of: "
            + ", ".join(_LABEL_CSVS)
        )
    df = pd.concat(frames, ignore_index=True).dropna(subset=["Image ID"])
    for col in _FINDING_COLS:
        df[col] = np.where(df[col] == "YES", 1, 0)
    agg = df.groupby("Image ID")[_FINDING_COLS].mean().reset_index()
    for col in _FINDING_COLS:
        agg[col] = (agg[col] >= 0.5).astype(int)
    return agg


class _ChxImageDataset(Dataset):
    """Yields ``(image, finding_bits[4])`` for a list of ``(path, findings)`` rows."""

    def __init__(self, rows: list[tuple[str, np.ndarray]], transform) -> None:
        self._rows = rows
        self._transform = transform

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int):
        path, findings = self._rows[idx]
        image = self._transform(Image.open(path).convert("RGB"))
        return image, torch.tensor(findings, dtype=torch.float32)


class ChxTask(Task):
    """ChestX-ray14 four-findings -> severity-code task."""

    n_classes: int = 5   # severity levels; unused by SPL directly
    in_channels: int = 3
    image_size: int = 224

    def __init__(self, root: str) -> None:
        from torchvision import transforms

        data_dir = Path(root) / "chx"
        img_dir = _find_image_dir(data_dir)
        has_csv = any((data_dir / name).exists() for name in _LABEL_CSVS)
        if not data_dir.exists() or img_dir is None or not has_csv:
            raise FileNotFoundError(
                f"CHX data not found at '{data_dir}'.\n"
                "Run 'make chx GCS_PROJECT=<billing-project>'."
            )

        agg = _majority_vote_findings(data_dir)
        agg = agg[agg["Image ID"].apply(lambda x: (img_dir / x).exists())].reset_index(drop=True)
        if len(agg) == 0:
            raise FileNotFoundError(
                f"No annotated CHX images present under '{img_dir}'. "
                "Run 'make chx GCS_PROJECT=<billing-project>'."
            )

        paths = [str(img_dir / img_id) for img_id in agg["Image ID"]]
        findings = agg[_FINDING_COLS].values.astype(np.float32)

        rng = np.random.RandomState(42)
        perm = rng.permutation(len(paths))
        n_test = int(len(perm) * _TEST_SPLIT)
        n_val = int(len(perm) * _VAL_SPLIT)
        test_idx = perm[:n_test]
        val_idx = perm[n_test : n_test + n_val]
        train_idx = perm[n_test + n_val :]

        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])
        eval_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])

        def rows(idxs, tf):
            return _ChxImageDataset([(paths[i], findings[i]) for i in idxs], tf)

        self._train_ds = rows(train_idx, train_transform)
        self._val_ds = rows(val_idx, eval_transform)
        self._test_ds = rows(test_idx, eval_transform)

    def train_loader(self, batch_size: int) -> DataLoader:
        return DataLoader(self._train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())

    def val_loader(self, batch_size: int) -> DataLoader:
        return DataLoader(self._val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=torch.cuda.is_available())

    def test_loader(self, batch_size: int) -> DataLoader:
        return DataLoader(self._test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=torch.cuda.is_available())
