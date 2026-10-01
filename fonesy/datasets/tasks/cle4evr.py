"""CLE4EVR (``cle4evr``): two-object CLEVR-style scenes (same shape and color).

Rendered by ``make cle4evr`` (Blender) into::

    fonesy/data/cle4evr/{train,val,test}/images/*.png
    fonesy/data/cle4evr/{train,val,test}/labels.json
        {"<file>": [obj1_shape, obj1_color, obj2_shape, obj2_color], ...}

with shape in (cube, sphere, cylinder) and color in (red, blue, green).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from fonesy.datasets.tasks.base import Task

_MEAN = (0.5, 0.5, 0.5)
_STD = (0.5, 0.5, 0.5)


class _CLE4evrSplit(Dataset):
    """Yields ``(image, [obj1_shape, obj1_color, obj2_shape, obj2_color])``."""

    def __init__(self, split_dir: Path, transform) -> None:
        labels_path = split_dir / "labels.json"
        self._images_dir = split_dir / "images"
        self._labels: dict[str, list[int]] = json.loads(labels_path.read_text())
        self._filenames = sorted(self._labels.keys())
        self._transform = transform

    def __len__(self) -> int:
        return len(self._filenames)

    def __getitem__(self, idx: int):
        filename = self._filenames[idx]
        image = Image.open(self._images_dir / filename).convert("RGB")
        image = self._transform(image)
        labels = torch.tensor(self._labels[filename], dtype=torch.long)
        return image, labels


class CLE4evrTask(Task):
    """CLE4EVR: two-object scenes, 4 raw label columns (shape/color x 2 objects)."""

    n_classes: int = 3      # unused directly by SPL (no separate classifier head)
    in_channels: int = 3
    image_size: int = 128

    def __init__(self, root: str) -> None:
        dataset_root = Path(root) / "cle4evr"
        if not dataset_root.exists():
            raise FileNotFoundError(
                f"CLE4EVR data not found at '{dataset_root}'.\n"
                "Run 'make cle4evr' (requires Blender)."
            )
        self._root = dataset_root
        self._transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
        ])

    def _split_dataset(self, split: str) -> _CLE4evrSplit:
        return _CLE4evrSplit(self._root / split, self._transform)

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
