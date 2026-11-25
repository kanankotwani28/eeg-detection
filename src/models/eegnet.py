"""EEGNet-8,2 (Lawhern et al., 2018) with GroupNorm instead of BatchNorm (DP- and FL-friendly:
no batch statistics). Input (B, 1, C, T) -> 2 logits. Max-norm constraints applied after each step."""
from __future__ import annotations

import torch
from torch import nn


class EEGNet(nn.Module):
    def __init__(self, n_channels=19, n_times=1024, n_classes=2, F1=8, D=2, F2=16, kernel_length=128,
                 dropout=0.5, n_groups=4, max_norm_spatial=1.0, max_norm_dense=0.25):
        super().__init__()
        self.max_norm_spatial, self.max_norm_dense = max_norm_spatial, max_norm_dense
        self.conv_temporal = nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False)
        self.norm1 = nn.GroupNorm(min(n_groups, F1), F1)
        self.conv_spatial = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False)
        self.norm2 = nn.GroupNorm(min(n_groups, F1 * D), F1 * D)
        self.act = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)
        self.conv_sep_depth = nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False)
        self.conv_sep_point = nn.Conv2d(F1 * D, F2, 1, bias=False)
        self.norm3 = nn.GroupNorm(min(n_groups, F2), F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout)
        with torch.no_grad():
            n_flat = self._features(torch.zeros(1, 1, n_channels, n_times)).shape[1]
        self.classifier = nn.Linear(n_flat, n_classes)

    def _features(self, x):
        x = self.norm1(self.conv_temporal(x))
        x = self.drop1(self.pool1(self.act(self.norm2(self.conv_spatial(x)))))
        x = self.norm3(self.conv_sep_point(self.conv_sep_depth(x)))
        x = self.drop2(self.pool2(self.act(x)))
        return x.flatten(1)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        return self.classifier(self._features(x))

    @torch.no_grad()
    def apply_max_norm(self):
        _renorm(self.conv_spatial.weight, self.max_norm_spatial)
        _renorm(self.classifier.weight, self.max_norm_dense)


def _renorm(w: torch.Tensor, max_norm: float):
    flat = w.view(w.shape[0], -1)
    norms = flat.norm(dim=1, keepdim=True).clamp(min=1e-12)
    flat.mul_(torch.clamp(max_norm / norms, max=1.0))


def build_eegnet(cfg: dict, n_channels=19, n_times=1024) -> EEGNet:
    keys = ("F1", "D", "F2", "kernel_length", "dropout", "n_groups", "max_norm_spatial", "max_norm_dense")
    return EEGNet(n_channels, n_times, 2, **{k: cfg[k] for k in keys if k in cfg})
