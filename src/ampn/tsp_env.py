"""Minimal TSP environment used to reproduce Kool 2019 figure 2."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


class TSPEnv:
    def __init__(self, n_cities: int, batch_size: int, device: str = "cpu") -> None:
        self.n = n_cities
        self.batch_size = batch_size
        self.device = torch.device(device)

    def reset(self, seed: int | None = None) -> Tensor:
        rng = np.random.default_rng(seed)
        self.coords = torch.from_numpy(
            rng.uniform(0, 1, size=(self.batch_size, self.n, 2))
        ).float().to(self.device)
        self.visited = torch.zeros(self.batch_size, self.n, dtype=torch.bool, device=self.device)
        self.last = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)
        self.cost = torch.zeros(self.batch_size, device=self.device)
        return self.coords

    def step(self, action: Tensor) -> Tensor:
        b = torch.arange(self.batch_size, device=self.device)
        d = (self.coords[b, action] - self.coords[b, self.last]).norm(dim=-1)
        self.cost = self.cost + d
        self.visited[b, action] = True
        self.last = action
        return d

    def feasibility_mask(self) -> Tensor:
        return self.visited
