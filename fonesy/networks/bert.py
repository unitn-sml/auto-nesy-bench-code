"""BertBackbone: a (frozen) BERT text encoder + small MLP projection, for
text-input SPL problems (CeBaB).

Mirrors ``MLPBackbone``'s role for BDD-OIA's precomputed features: the heavy
pretrained encoder is held fixed (a deliberate "not from scratch" exception,
exactly like ``FasterRCNNBackbone`` for cle4evr/kand-logic) and only the
lightweight projection MLP + the SPL head are trained jointly.

``.features`` accepts a ``(B, L, 2)`` integer tensor — ``[..., 0]`` is
``input_ids``, ``[..., 1]`` is ``attention_mask`` (the layout produced by
``fonesy.datasets.tasks.cebab.CeBaBTask``) — runs ``BertModel``, takes the
pooled ``[CLS]`` embedding and passes it through ``n_layers`` ``Linear +
ReLU`` blocks to ``hidden_dim``. ``.classifier`` is ``Linear(hidden_dim,
n_classes)`` (unused by SPL directly — only required by the
``.features``/``.classifier`` convention ``spl._get_feature_dim()`` relies
on).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from fonesy.networks.base import BaseCNN

_BERT_HIDDEN = 768


class _BertFeatures(nn.Module):
    """BERT (optionally frozen) -> [CLS] -> MLP projection to ``hidden_dim``."""

    def __init__(
        self,
        model_name: str,
        hidden_dim: int,
        n_layers: int,
        freeze_bert: bool,
    ) -> None:
        super().__init__()
        from transformers import BertModel

        self.bert = BertModel.from_pretrained(model_name)
        self.freeze_bert = freeze_bert
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad_(False)

        layers: list[nn.Module] = [nn.Linear(_BERT_HIDDEN, hidden_dim), nn.ReLU(inplace=True)]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        self.proj = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, 2) — [..., 0] input_ids, [..., 1] attention_mask.
        input_ids = x[..., 0].long()
        attention_mask = x[..., 1].long()
        ctx = torch.no_grad() if self.freeze_bert else torch.enable_grad()
        with ctx:
            out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]  # (B, 768)
        return self.proj(cls)


class BertBackbone(BaseCNN):
    """BERT text encoder (frozen by default) + MLP projection head.

    Constructor signature matches ``NetworkFactory.get_network``'s uniform
    ``(name, n_classes, in_channels, **kwargs)`` call; ``in_channels`` /
    ``image_size`` are unused placeholders.
    """

    def __init__(
        self,
        n_classes: int = 5,
        in_channels: int = 1,   # unused — text input, not an image
        image_size: int = 1,    # unused
        model_name: str = "bert-base-uncased",
        hidden_dim: int = 256,
        n_layers: int = 2,
        freeze_bert: bool = True,
    ) -> None:
        super().__init__(n_classes=n_classes, in_channels=in_channels, image_size=image_size)
        if n_layers not in (1, 2, 3):
            raise ValueError(f"BertBackbone supports 1-3 projection layers, got {n_layers}")
        self.features = _BertFeatures(model_name, hidden_dim, n_layers, freeze_bert)
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
