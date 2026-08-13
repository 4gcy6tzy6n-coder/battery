"""Aggregate probabilistic and point metrics over collected predictions.

All functions are pure NumPy and shape-validated. Quantile levels default to
``(0.05, 0.50, 0.95)``; the 90% interval is derived from the outer pair.
``crps_from_quantiles`` is the trapezoidal approximation
``CRPS = int_0^1 QL_tau d_tau`` and is unit-tested against an exact Gaussian
computation.
"""

from __future__ import annotations

import numpy as np

QUANTILES = np.array([0.05, 0.50, 0.95], dtype=float)
#: Dense experiment grid: every 0.05 level from 0.05 to 0.95 inclusive (19 levels).
EXPERIMENT_QUANTILES = np.arange(0.05, 1.0, 0.05)


def _as_1d(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _as_quantiles(pred_q: np.ndarray, levels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(pred_q, dtype=float)
    if pred.ndim != 2 or pred.shape[-1] != len(levels):
        raise ValueError(f"prediction must be [N, {len(levels)}]")
    if not np.isfinite(pred).all():
        raise ValueError("prediction must be finite")
    level = _as_1d(levels, "levels")
    if not ((level > 0) & (level < 1)).all():
        raise ValueError("levels must lie in (0, 1)")
    if not np.all(np.diff(level) > 0):
        raise ValueError("levels must be strictly increasing")
    return pred, level


def pinball_mean(pred_q: np.ndarray, target: np.ndarray, levels: np.ndarray) -> float:
    """Mean pinball loss over samples and levels."""
    pred, level = _as_quantiles(pred_q, levels)
    tgt = _as_1d(target, "target")
    if pred.shape[0] != tgt.shape[0]:
        raise ValueError("prediction and target must share the first dimension")
    error = tgt[:, None] - pred
    losses = np.maximum(level[None, :] * error, (level[None, :] - 1.0) * error)
    return float(losses.mean())


def crps_from_quantiles(pred_q: np.ndarray, target: np.ndarray, levels: np.ndarray) -> float:
    """CRPS approximated by trapezoid integration of the quantile-loss curve.

    Uses the identity ``CRPS(F, y) = 2 int_0^1 QL_tau d tau``. On the dense
    19-level grid the approximation is within ~5% of the exact Gaussian value.
    """
    pred, level = _as_quantiles(pred_q, levels)
    tgt = _as_1d(target, "target")
    if pred.shape[0] != tgt.shape[0]:
        raise ValueError("prediction and target must share the first dimension")
    error = tgt[:, None] - pred
    ql = np.maximum(level[None, :] * error, (level[None, :] - 1.0) * error)  # [N, K]
    if len(level) < 2:
        raise ValueError("at least two levels are required for CRPS")
    return float(2.0 * np.trapezoid(ql, level, axis=1).mean())


def interval_coverage(pred_lower: np.ndarray, pred_upper: np.ndarray, target: np.ndarray) -> float:
    """PICP: fraction of targets inside [lower, upper]."""
    lower = _as_1d(pred_lower, "lower")
    upper = _as_1d(pred_upper, "upper")
    tgt = _as_1d(target, "target")
    if not (lower.shape == upper.shape == tgt.shape):
        raise ValueError("lower, upper, and target must share shape")
    if np.any(lower > upper):
        raise ValueError("lower must not exceed upper")
    return float(np.mean((tgt >= lower) & (tgt <= upper)))


def interval_width(pred_lower: np.ndarray, pred_upper: np.ndarray) -> float:
    """Mean interval width (PINAW numerator, absolute units)."""
    lower = _as_1d(pred_lower, "lower")
    upper = _as_1d(pred_upper, "upper")
    if lower.shape != upper.shape:
        raise ValueError("lower and upper must share shape")
    if np.any(lower > upper):
        raise ValueError("lower must not exceed upper")
    return float(np.mean(upper - lower))


def quantile_ece(pred_q: np.ndarray, target: np.ndarray, levels: np.ndarray) -> float:
    """Mean absolute deviation between empirical and nominal coverage per level."""
    pred, level = _as_quantiles(pred_q, levels)
    tgt = _as_1d(target, "target")
    if pred.shape[0] != tgt.shape[0]:
        raise ValueError("prediction and target must share the first dimension")
    coverage = np.mean(tgt[:, None] <= pred, axis=0)
    return float(np.mean(np.abs(coverage - level)))


def mae_rmse(pred_median: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    """Return (MAE, RMSE, rRMSE) with rRMSE relative to the mean absolute target."""
    pred = _as_1d(pred_median, "median")
    tgt = _as_1d(target, "target")
    if pred.shape != tgt.shape:
        raise ValueError("median and target must share shape")
    error = pred - tgt
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    scale = float(np.mean(np.abs(tgt)))
    rrrmse = rmse / scale if scale > 0 else float("nan")
    return mae, rmse, rrrmse


def physical_consistency_error(
    soc_med: np.ndarray,
    soh_med: np.ndarray,
    log_tte_med: np.ndarray,
    q_ref_ah: np.ndarray,
    i_eff_60s: np.ndarray,
    minimum_current_a: float = 0.05,
) -> tuple[float, dict[str, float]]:
    """Absolute deviation between predicted log-TTE and the physical TTE formula."""
    soc = _as_1d(soc_med, "soc")
    soh = _as_1d(soh_med, "soh")
    tte = _as_1d(log_tte_med, "log_tte")
    q_ref = _as_1d(q_ref_ah, "q_ref")
    current = _as_1d(i_eff_60s, "i_eff")
    if not (soc.shape == soh.shape == tte.shape == q_ref.shape == current.shape):
        raise ValueError("all consistency inputs must share shape")
    active = current > minimum_current_a
    if not np.any(active):
        return float("nan"), {"active_fraction": 0.0}
    physical = soc * soh * q_ref / current * 3600.0
    error = np.abs(tte - np.log1p(np.clip(physical, 0.0, None)))[active]
    return float(np.mean(error)), {
        "active_fraction": float(np.mean(active)),
        "mean_abs_error": float(np.mean(error)),
        "median_abs_error": float(np.median(error)),
    }


def evaluate_state(
    pred_q: np.ndarray,
    target: np.ndarray,
    levels: np.ndarray = QUANTILES,
) -> dict[str, float]:
    """Compute the full metric bundle for one predicted state.

    The 90% interval uses the outer levels; the median is the middle level.
    All indices are derived from ``levels`` so arbitrary quantile counts work.
    """
    pred, _ = _as_quantiles(pred_q, levels)
    tgt = _as_1d(target, "target")
    if pred.shape[0] != tgt.shape[0]:
        raise ValueError("prediction and target must share the first dimension")
    median_index = (len(levels) - 1) // 2
    lower, median, upper = pred[:, 0], pred[:, median_index], pred[:, -1]
    mae, rmse, rrmse = mae_rmse(median, tgt)
    return {
        "pinball": pinball_mean(pred, tgt, levels),
        "crps": crps_from_quantiles(pred, tgt, levels),
        "mae": mae,
        "rmse": rmse,
        "rrmse": rrmse,
        "picp_90": interval_coverage(lower, upper, tgt),
        "pinaw_90": interval_width(lower, upper),
        "ece": quantile_ece(pred, tgt, levels),
    }
