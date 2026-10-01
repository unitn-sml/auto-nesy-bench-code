"""MNIST addition: two digits side by side -> their sum.

Batches are ``(images[B, 1, 28, 56], digits[B, 2], sum[B])``; MNIST is
downloaded by torchvision.
"""

from __future__ import annotations

import random

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms

from fonesy.datasets.tasks.base import Task

_MNIST_MEAN = (0.1307,)
_MNIST_STD = (0.3081,)

_VAL_SPLIT = 0.1


class MnistAdditionDataset(Dataset):
    """Paired MNIST dataset yielding ``(img_pair, digits, sum_label)`` triples.

    Parameters
    ----------
    dataset1, dataset2:
        Two map-style datasets of equal length whose items are
        ``(image, digit_label)`` pairs with ``image`` of shape ``(1, H, W)``.
        Pairs are formed element-wise: ``pairs[i] = (dataset1[i], dataset2[i])``.
        Pass two independently-shuffled views of the same underlying data so
        that each sample sees a different combination of digits.
    """

    def __init__(self, dataset1, dataset2) -> None:
        if len(dataset1) != len(dataset2):
            raise ValueError(
                f"dataset1 and dataset2 must have the same length, "
                f"got {len(dataset1)} and {len(dataset2)}"
            )
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.pairs = self._build_pairs()

    def _build_pairs(self) -> list:
        pairs = []
        for i in range(len(self.dataset1)):
            img1, label1 = self.dataset1[i]
            img2, label2 = self.dataset2[i]
            img_pair = torch.cat([img1, img2], dim=2)  # (1, 28, 56)
            digits = torch.tensor([label1, label2], dtype=torch.long)
            sum_label = int(label1) + int(label2)  # 0–18
            pairs.append((img_pair, digits, sum_label))
        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        return self.pairs[idx]


def _shuffled_subset(base_ds, indices: list, seed: int) -> Subset:
    """Return a ``Subset`` over *base_ds* with *indices* in a seeded-random order."""
    shuffled = indices.copy()
    random.Random(seed).shuffle(shuffled)
    return Subset(base_ds, shuffled)


class MnistAdditionTask(Task):
    """MNIST-Addition: paired digit images, 19 sum classes (0–18).

    For each split the base MNIST data is presented as two independently
    shuffled ``Subset`` views (seeds 0/1 for the first and second stream of
    each split).  Pairs are formed element-wise by ``MnistAdditionDataset``,
    so every sample combines a different pair of digits.

    Loaders yield 3-tuples ``(img_pair, digits, sum_label)``.  In the
    ``pretrain`` experiment the first element is used as input and the last
    as the classification target (sum class).
    """

    n_classes: int = 19   # sums 0–18
    in_channels: int = 1
    image_size: int = 28  # height; paired width is 56

    def __init__(self, root: str) -> None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_MNIST_MEAN, _MNIST_STD),
        ])
        full_ds = datasets.MNIST(root=root, train=True, download=True, transform=transform)

        val_size = int(_VAL_SPLIT * len(full_ds))
        train_size = len(full_ds) - val_size

        generator = torch.Generator().manual_seed(0)
        train_split, val_split = random_split(
            range(len(full_ds)),
            [train_size, val_size],
            generator=generator,
        )
        train_idx = list(train_split)
        val_idx = list(val_split)

        self._train_ds = MnistAdditionDataset(
            Subset(full_ds, train_idx),
            _shuffled_subset(full_ds, train_idx, seed=0),
        )
        self._val_ds = MnistAdditionDataset(
            Subset(full_ds, val_idx),
            _shuffled_subset(full_ds, val_idx, seed=1),
        )

        test_base = datasets.MNIST(root=root, train=False, download=True, transform=transform)
        test_idx = list(range(len(test_base)))
        self._test_ds = MnistAdditionDataset(
            Subset(test_base, test_idx),
            _shuffled_subset(test_base, test_idx, seed=2),
        )

    def train_loader(self, batch_size: int) -> DataLoader:
        return DataLoader(self._train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())

    def val_loader(self, batch_size: int) -> DataLoader:
        return DataLoader(self._val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())

    def test_loader(self, batch_size: int) -> DataLoader:
        return DataLoader(self._test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())
