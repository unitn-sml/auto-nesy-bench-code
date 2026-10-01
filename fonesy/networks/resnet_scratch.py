"""ResNet18Scratch: randomly-initialized ResNet-18 for larger (e.g. 224px) inputs.

Conv4 is sized for 28-32px inputs; this backbone is used for larger images.
``weights=None``: the backbone is trained jointly with the head from scratch.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18

from fonesy.networks.base import BaseCNN


class ResNet18Scratch(BaseCNN):
    """Randomly-initialized torchvision ResNet-18, split into ``.features``/``.classifier``.

    Matches the ``Conv4``/``LeNet5`` convention (``.features`` ends in a
    pooled ``(B, C, 1, 1)`` output, ``.classifier`` is ``Linear(C,
    n_classes)``) so ``fonesy/experiments/spl.py``'s ``_get_feature_dim()``
    keeps working unmodified.
    """

    def __init__(
        self,
        n_classes: int = 10,
        in_channels: int = 3,
        image_size: int = 224,
    ) -> None:
        super().__init__(n_classes=n_classes, in_channels=in_channels, image_size=image_size)
        backbone = resnet18(weights=None)
        if in_channels != 3:
            backbone.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
            backbone.avgpool,  # (B, 512, H', W') → (B, 512, 1, 1)
        )
        self.classifier = nn.Linear(backbone.fc.in_features, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(start_dim=1)
        return self.classifier(x)
