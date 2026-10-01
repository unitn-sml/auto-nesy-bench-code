"""Abstract base class for CNNs used in NeSy pre-training."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch.nn as nn


class BaseCNN(ABC, nn.Module):
    """Abstract base for all NeSy CNNs.

    Subclasses must implement ``forward`` and may override ``n_classes``,
    ``in_channels``, and ``image_size`` to reflect their architecture.
    """

    def __init__(self, n_classes: int, in_channels: int, image_size: int) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.in_channels = in_channels
        self.image_size = image_size

    @abstractmethod
    def forward(self, x):
        """Forward pass; returns logits of shape (B, n_classes)."""
        ...
