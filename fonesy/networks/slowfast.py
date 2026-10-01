"""SlowFast-50 video backbone (vendored from the official ROAD-R repository).

Source: https://github.com/EGiunchiglia/ROAD-R/blob/main/models/backbones/slowfast.py
(itself after "SlowFast Networks for Video Recognition", Feichtenhofer et al.
2019, and facebookresearch/SlowFast). The ``Bottleneck`` / ``SlowFast`` /
``slowfast50`` definitions below are copied verbatim except for two changes:

* the ``slow_full_span`` branch's hard-coded ``.cuda()`` is replaced with
  ``.to(input.device)`` (we never enable ``slow_full_span``, but keep it
  device-safe);
* ``alpha`` defaults to **2**, not 8. The vendored ``forward`` concatenates the
  slow/fast *feature maps* channel-wise after a single stride-2 temporal
  ``pool2`` on the fast path, which is shape-consistent only when the slow
  pathway runs at half the fast frame rate (``alpha == 2``). The ROAD-R repo
  drives this module from a YAML ``kwargs`` block that sets ``alpha``
  accordingly; here it is fixed so the module is self-contained.

``SlowFastBackbone`` wraps it in this project's ``.features`` / ``.classifier``
convention (see ``fonesy/networks/resnet_scratch.py``) so
``fonesy/experiments/spl.py`` can use it as ``--backbone slowfast``: ``.features``
maps a clip ``(B, 3, T, H, W)`` to a global-average-pooled ``(B, 2304, 1, 1, 1)``
tensor (2048 slow ++ 256 fast channels at the c5 stage), ``.classifier`` is
``Linear(2304, n_classes)``. Trained from scratch — the vendored file ships no
weight loader and no Kinetics-pretrained SlowFast checkpoint is publicly
available for it.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from fonesy.networks.base import BaseCNN

BN = nn.BatchNorm3d

_C5_CHANNELS = 512 * 4 + int(512 * 0.125) * 4  # slow (2048) ++ fast (256)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, head_conv=1):
        super(Bottleneck, self).__init__()
        if head_conv == 1:
            self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
            self.bn1 = BN(planes)
        elif head_conv == 3:
            self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(3, 1, 1), bias=False, padding=(1, 0, 0))
            self.bn1 = BN(planes)
        else:
            raise ValueError("Unsupported head_conv!")
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=(1, 3, 3), stride=(1, stride, stride),
            padding=(0, dilation, dilation), dilation=(1, dilation, dilation), bias=False)
        self.bn2 = BN(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BN(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        if downsample is not None:
            self.downsample_bn = BN(planes * 4)
        self.stride = stride

    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            res = self.downsample(x)
            res = self.downsample_bn(res)

        out = out + res
        out = self.relu(out)

        return out


class SlowFast(nn.Module):
    def __init__(self, block, layers, alpha=2, beta=0.125, fuse_only_conv=True, fuse_kernel_size=5, slow_full_span=False):
        super(SlowFast, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.slow_full_span = slow_full_span

        '''Fast Network'''
        self.fast_inplanes = int(64 * beta)
        self.fast_conv1 = nn.Conv3d(3, self.fast_inplanes, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False)
        self.fast_bn1 = BN(self.fast_inplanes)
        self.fast_relu = nn.ReLU(inplace=True)
        self.fast_maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.fast_res1 = self._make_layer_fast(block, int(64 * beta), layers[0], head_conv=3)
        self.fast_res2 = self._make_layer_fast(block, int(128 * beta), layers[1], stride=2, head_conv=3)
        self.fast_res3 = self._make_layer_fast(block, int(256 * beta), layers[2], stride=2, head_conv=3)
        self.fast_res4 = self._make_layer_fast(block, int(512 * beta), layers[3], head_conv=3, dilation=2)

        '''Slow Network'''
        self.slow_inplanes = 64
        self.slow_conv1 = nn.Conv3d(3, self.slow_inplanes, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        self.slow_bn1 = BN(self.slow_inplanes)
        self.slow_relu = nn.ReLU(inplace=True)
        self.slow_maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.slow_res1 = self._make_layer_slow(block, 64, layers[0], head_conv=1)
        self.slow_res2 = self._make_layer_slow(block, 128, layers[1], stride=2, head_conv=1)
        self.slow_res3 = self._make_layer_slow(block, 256, layers[2], stride=2, head_conv=3)
        self.slow_res4 = self._make_layer_slow(block, 512, layers[3], head_conv=3, dilation=2)

        '''Lateral Connections'''
        fuse_padding = fuse_kernel_size // 2
        fuse_kwargs = {'kernel_size': (fuse_kernel_size, 1, 1), 'stride': (alpha, 1, 1), 'padding': (fuse_padding, 0, 0), 'bias': False}
        if fuse_only_conv:
            def fuse_func(in_channels, out_channels):
                return nn.Conv3d(in_channels, out_channels, **fuse_kwargs)
        else:
            def fuse_func(in_channels, out_channels):
                return nn.Sequential(
                    nn.Conv3d(in_channels, out_channels, **fuse_kwargs),
                    BN(out_channels),
                    nn.ReLU(inplace=True)
                )
        self.Tconv1 = fuse_func(int(64 * beta), int(128 * beta))
        self.Tconv2 = fuse_func(int(256 * beta), int(512 * beta))
        self.Tconv3 = fuse_func(int(512 * beta), int(1024 * beta))
        self.Tconv4 = fuse_func(int(1024 * beta), int(2048 * beta))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))

    def _upsample(self, x, y):
        _, _, t, h, w = y.size()
        x_upsampled = F.interpolate(x, [t, h, w], mode='nearest')
        return x_upsampled

    def forward(self, input):
        fast, Tc = self.FastPath(input)

        if self.slow_full_span:
            slow_input = torch.index_select(
                input,
                2,
                torch.linspace(
                    0,
                    input.shape[2] - 1,
                    input.shape[2] // self.alpha,
                ).long().to(input.device),
            )
        else:
            slow_input = input[:, :, ::self.alpha, :, :]
        slow = self.SlowPath(slow_input, Tc)

        fast[0] = self.pool2(fast[0])
        fast[1] = self.pool2(fast[1])
        fast[2] = self.pool2(fast[2])

        outFeat = []
        for sitem, fitem in zip(slow, fast):
            outFeat.append(torch.cat((sitem, fitem), 1))
        return outFeat

    def SlowPath(self, input, Tc):
        x = self.slow_conv1(input)
        x = self.slow_bn1(x)
        x = self.slow_relu(x)
        x = self.slow_maxpool(x)
        x = torch.cat([x, Tc[0]], dim=1)
        x = self.slow_res1(x)
        x = torch.cat([x, Tc[1]], dim=1)
        c3 = self.slow_res2(x)
        x = torch.cat([c3, Tc[2]], dim=1)
        c4 = self.slow_res3(x)
        x = torch.cat([c4, Tc[3]], dim=1)
        c5 = self.slow_res4(x)
        return [c3, c4, c5]

    def FastPath(self, input):
        x = self.fast_conv1(input)
        x = self.fast_bn1(x)
        x = self.fast_relu(x)
        x = self.fast_maxpool(x)
        Tc1 = self.Tconv1(x)
        x = self.fast_res1(x)
        Tc2 = self.Tconv2(x)
        c3 = self.fast_res2(x)
        Tc3 = self.Tconv3(c3)
        c4 = self.fast_res3(c3)
        Tc4 = self.Tconv4(c4)
        c5 = self.fast_res4(c4)
        return [c3, c4, c5], [Tc1, Tc2, Tc3, Tc4]

    def _make_layer_fast(self, block, planes, blocks, stride=1, head_conv=1, dilation=1):
        downsample = None
        if stride != 1 or self.fast_inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.fast_inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=(1, stride, stride),
                    bias=False
                )
            )

        layers = []
        layers.append(block(self.fast_inplanes, planes, stride, downsample, dilation=dilation, head_conv=head_conv))
        self.fast_inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.fast_inplanes, planes, dilation=dilation, head_conv=head_conv))

        return nn.Sequential(*layers)

    def _make_layer_slow(self, block, planes, blocks, stride=1, head_conv=1, dilation=1):
        downsample = None
        fused_inplanes = self.slow_inplanes + int(self.slow_inplanes * self.beta) * 2
        if stride != 1 or fused_inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    fused_inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=(1, stride, stride),
                    bias=False
                )
            )

        layers = []
        layers.append(block(fused_inplanes, planes, stride, downsample, dilation=dilation, head_conv=head_conv))
        self.slow_inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.slow_inplanes, planes, dilation=dilation, head_conv=head_conv))

        return nn.Sequential(*layers)


def slowfast50(**kwargs):
    """Constructs a SlowFast-50 model."""
    return SlowFast(Bottleneck, [3, 4, 6, 3], **kwargs)


class _SlowFastFeatures(nn.Module):
    """SlowFast-50 + global spatiotemporal average pool over its c5 feature map.

    Input  : clip ``(B, 3, T, H, W)`` (T even; T >= 2).
    Output : ``(B, 2304, 1, 1, 1)`` — flattens to a 2304-d vector for the SPL head.
    """

    def __init__(self, alpha: int = 2) -> None:
        super().__init__()
        self.net = slowfast50(alpha=alpha)
        self.pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        c5 = self.net(clip)[-1]        # (B, 2304, T', H', W')
        return self.pool(c5)           # (B, 2304, 1, 1, 1)


class SlowFastBackbone(BaseCNN):
    """SlowFast-50 video backbone in this project's ``.features``/``.classifier`` convention.

    ``image_size`` is accepted for interface parity with the other backbones
    but unused (SlowFast is fully convolutional). ``in_channels`` must be 3.
    """

    def __init__(self, n_classes: int = 2, in_channels: int = 3, image_size: int = 96, alpha: int = 2) -> None:
        super().__init__(n_classes=n_classes, in_channels=in_channels, image_size=image_size)
        if in_channels != 3:
            raise ValueError("SlowFastBackbone requires 3-channel RGB clips")
        self.features = _SlowFastFeatures(alpha=alpha)
        self.classifier = nn.Linear(_C5_CHANNELS, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(start_dim=1)
        return self.classifier(x)
