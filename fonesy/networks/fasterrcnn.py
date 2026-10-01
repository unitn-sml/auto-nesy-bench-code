"""FasterRCNNBackbone: COCO-pretrained Faster-RCNN ResNet50-FPN as a feature extractor.

Unlike every other backbone in this pipeline (trained fully from scratch), a
freshly-initialized Faster-RCNN backbone provides no more value than the
existing ``ResNet18Scratch`` — Faster-RCNN's value is specifically its
COCO-pretrained detection-aware features. This mirrors how BDD-OIA already
uses *pretrained* (not from-scratch) Faster-RCNN features in this same
pipeline: a deliberate, explicit exception to "everything from scratch," not
a violation of it. The backbone is still fine-tuned *jointly* with the SPL
head (not frozen), like every other backbone here — the exception is only
the pretrained initialization.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from fonesy.networks.base import BaseCNN

# BackboneWithFPN's 5 feature-pyramid levels ('0','1','2','3','pool'), each
# 256 channels — global-average-pooled and concatenated into one flat vector.
_N_FPN_LEVELS = 5
_FPN_CHANNELS = 256


class _FasterRCNNFeatures(nn.Module):
    """Wraps Faster-RCNN's FPN backbone, pooling its multi-scale output into one vector."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature_maps = self.backbone(x)  # OrderedDict of (B, 256, H_i, W_i)
        pooled = [self.pool(fm).flatten(1) for fm in feature_maps.values()]
        return torch.cat(pooled, dim=1)  # (B, 5*256) = (B, 1280)


class FasterRCNNBackbone(BaseCNN):
    """COCO-pretrained Faster-RCNN ResNet50-FPN backbone, fine-tuned jointly with the head.

    ``.features`` outputs a ``(B, 1280)`` vector (5 FPN levels x 256 channels,
    globally pooled and concatenated). ``.classifier`` follows the standard
    convention (unused by SPL directly, keeps ``_get_feature_dim()`` working).
    """

    def __init__(
        self,
        n_classes: int = 10,
        in_channels: int = 3,
        image_size: int = 224,
    ) -> None:
        super().__init__(n_classes=n_classes, in_channels=in_channels, image_size=image_size)
        if in_channels != 3:
            raise ValueError(
                f"FasterRCNNBackbone requires in_channels=3 (COCO-pretrained "
                f"RGB weights), got {in_channels}"
            )
        detection_model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
        self.features = _FasterRCNNFeatures(detection_model.backbone)
        self.classifier = nn.Linear(_N_FPN_LEVELS * _FPN_CHANNELS, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)
