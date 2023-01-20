"""Pointer decoder for TSP / VRP (Kool 2019)."""

from __future__ import annotations

import math
import torch
from torch import Tensor, nn


class PointerDecoder(nn.Module):
    def __init__(self, d: int = 128, h: int = 8, tanh_clip: float = 10.0) -> None:
        super().__init__()
        self.tanh_clip = tanh_clip
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.glimpse = nn.MultiheadAttention(d, h, batch_first=True)
        self.context_proj = nn.Linear(d * 2, d, bias=False)
        self.d = d

    def forward(self, node_emb: Tensor, last_emb: Tensor, mask: Tensor) -> Tensor:
        graph = node_emb.mean(dim=1)
        ctx = self.context_proj(torch.cat([graph, last_emb], dim=-1)).unsqueeze(1)
        ctx, _ = self.glimpse(ctx, node_emb, node_emb, key_padding_mask=mask)
        q = self.q_proj(ctx)
        k = self.k_proj(node_emb)
        logits = (q @ k.transpose(-1, -2)).squeeze(1) / math.sqrt(self.d)
        logits = self.tanh_clip * torch.tanh(logits)
        return logits.masked_fill(mask, float("-inf"))
