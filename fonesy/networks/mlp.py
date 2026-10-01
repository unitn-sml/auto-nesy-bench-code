"""MLPBackbone: a small 2-3 layer MLP feature extractor for pre-extracted
feature-vector inputs (e.g. BDD-OIA's precomputed 2048-d Faster-RCNN
embeddings) — no image/convolutional structure needed, unlike Conv4/LeNet5/
ResNet18Scratch.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from fonesy.networks.base import BaseCNN


class MLPBackbone(BaseCNN):
    """Small MLP for flat feature-vector inputs (not images).

    ``.features``: ``n_layers`` ``Linear(..., hidden_dim) + ReLU`` blocks
    (``in_dim -> hidden_dim`` then ``hidden_dim -> hidden_dim`` for the
    remaining layers). ``.classifier``: ``Linear(hidden_dim, n_classes)``
    (unused by SPL directly — only required by ``NetworkFactory.get_network``'s
    uniform signature; kept for the ``.features``/``.classifier`` convention
    ``fonesy.experiments.spl._get_feature_dim()`` relies on).
    """

    def __init__(
        self,
        n_classes: int = 10,
        in_channels: int = 1,   # unused — flat-vector input, not an image
        image_size: int = 1,    # unused
        in_dim: int = 2048,
        hidden_dim: int = 512,
        n_layers: int = 3,
    ) -> None:
        super().__init__(n_classes=n_classes, in_channels=in_channels, image_size=image_size)
        if n_layers not in (2, 3):
            raise ValueError(f"MLPBackbone supports 2 or 3 layers, got {n_layers}")

        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True)]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)
