"""Strictly validated quantile-regression loss."""

from __future__ import annotations

import torch


def pinball_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    quantiles: torch.Tensor,
) -> torch.Tensor:
    """Return the mean pinball loss after rejecting unsafe broadcasting."""
    if pred.ndim < 1 or pred.shape[-1] != 3:
        raise ValueError("prediction last dimension must contain three quantiles")
    if target.ndim == pred.ndim and target.shape[-1:] == (1,):
        target = target.squeeze(-1)
    if target.shape != pred.shape[:-1]:
        raise ValueError(
            f"target shape {tuple(target.shape)} must equal {tuple(pred.shape[:-1])}"
        )
    if quantiles.ndim != 1 or quantiles.numel() != 3:
        raise ValueError("quantiles must be a one-dimensional tensor of length three")
    if not torch.isfinite(quantiles).all():
        raise ValueError("quantiles must be finite")
    if not ((quantiles > 0) & (quantiles < 1)).all():
        raise ValueError("quantiles must lie strictly inside (0, 1)")
    if not (torch.diff(quantiles) > 0).all():
        raise ValueError("quantiles must be strictly increasing")

    target = target.to(device=pred.device, dtype=pred.dtype)
    levels = quantiles.to(device=pred.device, dtype=pred.dtype)
    error = target.unsqueeze(-1) - pred
    loss = torch.maximum(levels * error, (levels - 1.0) * error)
    return loss.mean()
