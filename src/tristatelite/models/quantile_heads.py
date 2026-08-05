"""Ordered quantile heads whose parameterization guarantees valid outputs."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

QUANTILES = (0.05, 0.50, 0.95)


class BoundedQuantileHead(nn.Module):
    """Produce ordered lower, median, and upper quantiles inside ``[0, 1]``."""

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        if in_dim <= 0:
            raise ValueError("in_dim must be positive")
        self.projection = nn.Linear(in_dim, 3)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        median_raw, lower_raw, upper_raw = self.projection(hidden).unbind(dim=-1)
        q50 = torch.sigmoid(median_raw)
        q05 = q50 * (1.0 - torch.sigmoid(lower_raw))
        q95 = q50 + (1.0 - q50) * torch.sigmoid(upper_raw)
        return torch.stack((q05, q50, q95), dim=-1)


class PositiveQuantileHead(nn.Module):
    """Produce ordered non-negative lower, median, and upper quantiles."""

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        if in_dim <= 0:
            raise ValueError("in_dim must be positive")
        self.projection = nn.Linear(in_dim, 3)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        median_raw, lower_raw, upper_raw = self.projection(hidden).unbind(dim=-1)
        q50 = F.softplus(median_raw)
        q05 = q50 * torch.sigmoid(lower_raw)
        q95 = q50 + F.softplus(upper_raw)
        return torch.stack((q05, q50, q95), dim=-1)
