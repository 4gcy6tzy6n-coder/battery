"""Ordered quantile heads whose parameterization guarantees valid outputs.

Both heads support an arbitrary odd number of quantiles. The median is the
middle output; lower quantiles are produced by multiplying the median by
sigmoid factors (decreasing, bounded by the median from above), and upper
quantiles by additive increments. For ``num_quantiles=3`` the construction is
bit-for-bit identical to the original Phase 2B formulas.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

QUANTILES = (0.05, 0.50, 0.95)


def _validate_num_quantiles(num_quantiles: int) -> None:
    if num_quantiles < 3 or num_quantiles % 2 == 0:
        raise ValueError("num_quantiles must be an odd integer of at least three")
    if num_quantiles > 101:
        raise ValueError("num_quantiles is unreasonably large")


class BoundedQuantileHead(nn.Module):
    """Produce ordered lower, median, and upper quantiles inside ``[0, 1]``."""

    def __init__(self, in_dim: int, num_quantiles: int = 3) -> None:
        super().__init__()
        if in_dim <= 0:
            raise ValueError("in_dim must be positive")
        _validate_num_quantiles(num_quantiles)
        self.num_quantiles = num_quantiles
        self.median_index = num_quantiles // 2
        self.projection = nn.Linear(in_dim, num_quantiles)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        raw = self.projection(hidden)
        middle = self.median_index
        outputs: list[torch.Tensor] = [None] * self.num_quantiles  # type: ignore[list-item]
        q50 = torch.sigmoid(raw[..., middle])
        outputs[middle] = q50
        value = q50
        for index in range(middle - 1, -1, -1):
            value = value * torch.sigmoid(raw[..., index])
            outputs[index] = value
        value = q50
        for index in range(middle + 1, self.num_quantiles):
            value = value + (1.0 - value) * torch.sigmoid(raw[..., index])
            outputs[index] = value
        return torch.stack(outputs, dim=-1)


class PositiveQuantileHead(nn.Module):
    """Produce ordered non-negative lower, median, and upper quantiles."""

    def __init__(self, in_dim: int, num_quantiles: int = 3) -> None:
        super().__init__()
        if in_dim <= 0:
            raise ValueError("in_dim must be positive")
        _validate_num_quantiles(num_quantiles)
        self.num_quantiles = num_quantiles
        self.median_index = num_quantiles // 2
        self.projection = nn.Linear(in_dim, num_quantiles)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        raw = self.projection(hidden)
        middle = self.median_index
        outputs: list[torch.Tensor] = [None] * self.num_quantiles  # type: ignore[list-item]
        q50 = F.softplus(raw[..., middle])
        outputs[middle] = q50
        value = q50
        for index in range(middle - 1, -1, -1):
            value = value * torch.sigmoid(raw[..., index])
            outputs[index] = value
        value = q50
        for index in range(middle + 1, self.num_quantiles):
            value = value + F.softplus(raw[..., index])
            outputs[index] = value
        return torch.stack(outputs, dim=-1)
