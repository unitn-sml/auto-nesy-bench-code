"""CIFAR-10 (``cifar10``): image -> 7 visual concepts -> 8 classes.

Automobile/truck and dog/horse are merged, so the concepts determine the
class. The concepts are a function of the class, hence items are
``(image[3, 32, 32], merged_class_index)``; CIFAR-10 is downloaded by
torchvision.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms

from fonesy.datasets.tasks.base import Task

_CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR_STD = (0.2470, 0.2435, 0.2616)

_TRAIN_SIZE = 45_000
_VAL_SIZE = 5_000

# torchvision CIFAR-10 label index -> cifar10 oracle's merged 8-way label index.
# CIFAR-10:  0 airplane 1 automobile 2 bird 3 cat 4 deer 5 dog 6 frog 7 horse 8 ship 9 truck
# oracle:    0 airplane 1 car_or_truck 2 bird 3 frog 4 deer 5 cat 6 dog_or_horse 7 ship
CIFAR10_TO_MERGED: dict[int, int] = {
    0: 0,  # airplane
    1: 1,  # automobile -> car_or_truck
    2: 2,  # bird
    3: 5,  # cat
    4: 4,  # deer
    5: 6,  # dog -> dog_or_horse
    6: 3,  # frog
    7: 6,  # horse -> dog_or_horse
    8: 7,  # ship
    9: 1,  # truck -> car_or_truck
}


class _MergedLabelDataset(Dataset):
    """Yield ``(image, merged_class_index[1])`` (length-1 long tensor)."""

    def __init__(self, base: Dataset) -> None:
        self._base = base

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int):
        image, label = self._base[idx]
        return image, torch.tensor([CIFAR10_TO_MERGED[int(label)]], dtype=torch.long)


class Cifar10ConceptTask(Task):
    """CIFAR-10 with the 8-way merged label set of the ``cifar10`` oracle."""

    n_classes: int = 8
    in_channels: int = 3
    image_size: int = 32

    def __init__(self, root: str) -> None:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
        ])
        eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
        ])
        full_aug = datasets.CIFAR10(root=root, train=True, download=True, transform=train_transform)
        full_eval = datasets.CIFAR10(root=root, train=True, download=True, transform=eval_transform)

        generator = torch.Generator().manual_seed(0)
        train_idx, val_idx = random_split(
            range(len(full_aug)), [_TRAIN_SIZE, _VAL_SIZE], generator=generator
        )
        self._train_ds = _MergedLabelDataset(Subset(full_aug, list(train_idx)))
        self._val_ds = _MergedLabelDataset(Subset(full_eval, list(val_idx)))
        self._test_ds = _MergedLabelDataset(
            datasets.CIFAR10(root=root, train=False, download=True, transform=eval_transform)
        )

    def train_loader(self, batch_size: int) -> DataLoader:
        return DataLoader(self._train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())

    def val_loader(self, batch_size: int) -> DataLoader:
        return DataLoader(self._val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())

    def test_loader(self, batch_size: int) -> DataLoader:
        return DataLoader(self._test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())
