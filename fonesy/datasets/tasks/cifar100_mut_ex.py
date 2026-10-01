"""CIFAR-100 (``cifar100``): exactly one of 100 class indicators is true.

Items are ``(image[3, 32, 32], class_index)``; CIFAR-100 is downloaded by
torchvision.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

from fonesy.datasets.tasks.base import Task
from fonesy.datasets.tasks.fashion_mut_ex import _LabelAsConceptDataset

_CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
_CIFAR100_STD = (0.2673, 0.2564, 0.2762)

_TRAIN_SIZE = 45_000
_VAL_SIZE = 5_000


class Cifar100MutExTask(Task):
    """CIFAR-100, 100 mutually-exclusive classes, ``3x32x32``."""

    n_classes: int = 100
    in_channels: int = 3
    image_size: int = 32

    def __init__(self, root: str) -> None:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR100_MEAN, _CIFAR100_STD),
        ])
        eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR100_MEAN, _CIFAR100_STD),
        ])
        full_aug = datasets.CIFAR100(root=root, train=True, download=True, transform=train_transform)
        full_eval = datasets.CIFAR100(root=root, train=True, download=True, transform=eval_transform)

        generator = torch.Generator().manual_seed(0)
        train_idx, val_idx = random_split(
            range(len(full_aug)), [_TRAIN_SIZE, _VAL_SIZE], generator=generator
        )
        self._train_ds = _LabelAsConceptDataset(Subset(full_aug, list(train_idx)))
        self._val_ds = _LabelAsConceptDataset(Subset(full_eval, list(val_idx)))
        self._test_ds = _LabelAsConceptDataset(
            datasets.CIFAR100(root=root, train=False, download=True, transform=eval_transform)
        )

    def train_loader(self, batch_size: int) -> DataLoader:
        return DataLoader(self._train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())

    def val_loader(self, batch_size: int) -> DataLoader:
        return DataLoader(self._val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())

    def test_loader(self, batch_size: int) -> DataLoader:
        return DataLoader(self._test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())
