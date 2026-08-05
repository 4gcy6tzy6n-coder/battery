"""Load-conditioned physical consistency for SOC, SOH, and TTE medians."""

from __future__ import annotations

import torch


def physics_consistency_loss(
    soc_median: torch.Tensor,
    soh_median: torch.Tensor,
    log_tte_median: torch.Tensor,
    q_ref_ah: torch.Tensor,
    i_eff_60s: torch.Tensor,
    current_cv_60s: torch.Tensor,
    *,
    minimum_current_a: float = 0.05,
    maximum_tte_s: float = 604800.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compare predicted log-TTE with a stable-load capacity estimate."""
    if minimum_current_a <= 0:
        raise ValueError("minimum_current_a must be positive")
    if maximum_tte_s <= 0:
        raise ValueError("maximum_tte_s must be positive")

    inputs = (
        soc_median,
        soh_median,
        log_tte_median,
        q_ref_ah,
        i_eff_60s,
        current_cv_60s,
    )
    if any(value.shape != log_tte_median.shape for value in inputs):
        raise ValueError("all consistency inputs must have the same shape")

    device = log_tte_median.device
    dtype = log_tte_median.dtype
    soc, soh, predicted, q_ref, current, current_cv = (
        value.to(device=device, dtype=dtype) for value in inputs
    )
    active = torch.ones_like(predicted, dtype=torch.bool)
    for value in (soc, soh, predicted, q_ref, current, current_cv):
        active = active & torch.isfinite(value)
    active = active & (current > minimum_current_a)

    zero = torch.zeros_like(predicted)
    safe_soc = torch.where(active, soc, zero)
    safe_soh = torch.where(active, soh, zero)
    safe_q_ref = torch.where(active, q_ref, zero)
    safe_current = torch.where(active, current, torch.ones_like(current))
    safe_cv = torch.where(active, current_cv, zero)
    physical_tte_s = (safe_soc * safe_soh * safe_q_ref / safe_current * 3600.0).clamp(
        min=0.0, max=maximum_tte_s
    )
    weight = torch.exp(-2.0 * safe_cv.clamp_min(0.0))
    error = torch.abs(predicted - torch.log1p(physical_tte_s))
    denominator = active.sum().clamp_min(1).to(dtype=dtype)
    loss = torch.where(active, weight * error, zero).sum() / denominator
    loss = loss + predicted.sum() * 0.0

    with torch.no_grad():
        if active.any():
            active_physical_tte = physical_tte_s[active]
            diagnostics = {
                "active_fraction": active.to(dtype=dtype).mean(),
                "mean_weight": weight[active].mean(),
                "median_physical_tte_s": active_physical_tte.median(),
            }
        else:
            scalar_zero = torch.zeros((), device=device, dtype=dtype)
            diagnostics = {
                "active_fraction": scalar_zero,
                "mean_weight": scalar_zero,
                "median_physical_tte_s": scalar_zero,
            }
    return loss, {name: value.detach() for name, value in diagnostics.items()}
