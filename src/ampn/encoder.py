"""Attention-model encoder (Kool et al. 2019)."""

from __future__ import annotations

import math
import torch
from torch import Tensor, nn


class MHA(nn.Module):
    def __init__(self, d: int, h: int) -> None:
        super().__init__()
        assert d % h == 0
        self.h = h
        self.dh = d // h
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, n, d)
        return self.out(out)


class AMBlock(nn.Module):
    def __init__(self, d: int, h: int) -> None:
        super().__init__()
        self.attn = MHA(d, h)
        self.norm1 = nn.BatchNorm1d(d)
        self.ff = nn.Sequential(nn.Linear(d, d * 4), nn.GELU(), nn.Linear(d * 4, d))
        self.norm2 = nn.BatchNorm1d(d)

    def forward(self, x: Tensor) -> Tensor:
        # batch-norm over feature dim (Kool 2019 uses batch-norm, not layer-norm)
        h = self.norm1((x + self.attn(x)).transpose(1, 2)).transpose(1, 2)
        out = self.norm2((h + self.ff(h)).transpose(1, 2)).transpose(1, 2)
        return out


class AttentionModelEncoder(nn.Module):
    def __init__(self, in_dim: int, d: int = 128, h: int = 8, n_layers: int = 3) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_dim, d)
        self.blocks = nn.ModuleList([AMBlock(d, h) for _ in range(n_layers)])

    def forward(self, x: Tensor) -> Tensor:
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        return h
