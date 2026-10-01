"""Abstract base class for visual pre-training tasks."""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch.utils.data import DataLoader


class Task(ABC):
    """Abstract visual classification task.

    Class attributes
    ----------------
    n_classes : int
        Number of output classes.
    in_channels : int
        Number of image channels (1 for grayscale, 3 for RGB).
    image_size : int
        Spatial dimension of a square image.
    """

    n_classes: int
    in_channels: int
    image_size: int

    @abstractmethod
    def train_loader(self, batch_size: int) -> DataLoader:
        """Return a DataLoader for the training split (with augmentation)."""
        ...

    @abstractmethod
    def val_loader(self, batch_size: int) -> DataLoader:
        """Return a DataLoader for the validation split (no augmentation)."""
        ...

    @abstractmethod
    def test_loader(self, batch_size: int) -> DataLoader:
        """Return a DataLoader for the test split."""
        ...
