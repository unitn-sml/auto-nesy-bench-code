"""Fashion-MNIST (``fashion``): exactly one of 10 class indicators is true.

Items are ``(image[1, 28, 28], class_index)``; Fashion-MNIST is downloaded by
torchvision.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms

from fonesy.datasets.tasks.base import Task

_FMNIST_MEAN = (0.2860,)
_FMNIST_STD = (0.3530,)

_VAL_SPLIT = 0.1


class _LabelAsConceptDataset(Dataset):
    """Wrap an ``(image, label)`` dataset to yield ``(image, label[1])`` where
    the label is a length-1 long tensor (so the batch is ``(B, 1)``, the shape
    ``ProblemSpec._make_onehot_concept_encoder`` expects)."""

    def __init__(self, base: Dataset) -> None:
        self._base = base

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int):
        image, label = self._base[idx]
        return image, torch.tensor([int(label)], dtype=torch.long)


class FashionMutExTask(Task):
    """Fashion-MNIST, 10 mutually-exclusive classes, ``1x28x28``."""

    n_classes: int = 10
    in_channels: int = 1
    image_size: int = 28

    def __init__(self, root: str) -> None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_FMNIST_MEAN, _FMNIST_STD),
        ])
        full_ds = datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)

        val_size = int(_VAL_SPLIT * len(full_ds))
        train_size = len(full_ds) - val_size
        generator = torch.Generator().manual_seed(0)
        train_idx, val_idx = random_split(
            range(len(full_ds)), [train_size, val_size], generator=generator
        )
        self._train_ds = _LabelAsConceptDataset(Subset(full_ds, list(train_idx)))
        self._val_ds = _LabelAsConceptDataset(Subset(full_ds, list(val_idx)))

        test_base = datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)
        self._test_ds = _LabelAsConceptDataset(test_base)

    def train_loader(self, batch_size: int) -> DataLoader:
        return DataLoader(self._train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())

    def val_loader(self, batch_size: int) -> DataLoader:
        return DataLoader(self._val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())

    def test_loader(self, batch_size: int) -> DataLoader:
        return DataLoader(self._test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())
