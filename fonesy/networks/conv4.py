"""Conv4: four-block CNN for CIFAR-10 (32×32×3)."""

from __future__ import annotations

import torch
import torch.nn as nn

from fonesy.networks.base import BaseCNN


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class Conv4(BaseCNN):
    """4-block CNN for CIFAR-10 (and paired MNIST images).

    Architecture: 4× (Conv3×3 / BN / ReLU / MaxPool2) → AdaptiveAvgPool → Linear(64, n_classes).
    AdaptiveAvgPool2d(1) collapses any spatial size to (B, 64, 1, 1), making Conv4
    input-size agnostic — it handles both square (32×32) and non-square (28×56) inputs.
    """

    def __init__(
        self,
        n_classes: int = 10,
        in_channels: int = 3,
        image_size: int = 32,
    ) -> None:
        super().__init__(n_classes=n_classes, in_channels=in_channels, image_size=image_size)
        self.features = nn.Sequential(
            _conv_block(in_channels, 64),
            _conv_block(64, 64),
            _conv_block(64, 64),
            _conv_block(64, 64),
            nn.AdaptiveAvgPool2d(1),  # (B, 64, H', W') → (B, 64, 1, 1)
        )
        self.classifier = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(start_dim=1)
        return self.classifier(x)
