"""Post-hoc empirical quantile recalibration (conformal-style).

Fits, on a held-out calibration set, the empirical mapping from nominal to
actual coverage per quantile level, then remaps test quantiles so empirical
coverage matches nominal. This is distribution-free and only requires
predictions from one model, so it is a standard post-processing step.
"""

from __future__ import annotations

import numpy as np
import scipy.interpolate


def fit_calibration(
    pred_q: np.ndarray, targets: np.ndarray, levels: np.ndarray
) -> dict[str, np.ndarray]:
    """Return the empirical coverage map ``coverage[level] = P(y <= q_level)``.

    ``pred_q`` is ``[N, K]``; ``targets`` is ``[N]``; ``levels`` is the ``[K]``
    nominal grid. Coverage must be strictly increasing for the inverse map to
    exist, so we project onto a monotone staircase via ``np.maximum.accumulate``.
    """
    pred = np.asarray(pred_q, dtype=float)
    tgt = np.asarray(targets, dtype=float)
    if pred.ndim != 2 or pred.shape[1] != len(levels):
        raise ValueError("pred_q must be [N, K] matching levels")
    if pred.shape[0] != len(tgt):
        raise ValueError("pred_q and targets must share the first dimension")
    coverage = np.mean(tgt[:, None] <= pred, axis=0)
    # Enforce monotonicity (finite-sample coverage may dip locally).
    coverage = np.maximum.accumulate(np.minimum.accumulate(coverage[::-1])[::-1])
    return {"levels": np.asarray(levels, dtype=float), "coverage": coverage}


def apply_calibration(
    pred_q: np.ndarray,
    levels: np.ndarray,
    calibration: dict[str, np.ndarray],
    *,
    domain: tuple[float | None, float | None] | None = None,
) -> np.ndarray:
    """Remap test quantiles so nominal coverage is achieved on the calibration set.

    ``domain`` optionally clips the result (e.g. ``(0.0, 1.0)`` for bounded
    SOC/SOH, ``(0.0, None)`` for non-negative log-TTE). Under-dispersed
    forecasts require extrapolation beyond the observed quantiles, which is
    applied linearly and then clipped to the domain.
    """
    pred = np.asarray(pred_q, dtype=float)
    levels = np.asarray(levels, dtype=float)
    cal_levels = np.asarray(calibration["levels"], dtype=float)
    cal_coverage = np.asarray(calibration["coverage"], dtype=float)
    if pred.ndim != 2 or pred.shape[1] != len(levels):
        raise ValueError("pred_q must be [N, K] matching levels")

    # Desired level tau -> source level L with coverage(L) == tau (extrapolated).
    source_levels = scipy.interpolate.interp1d(
        cal_coverage, cal_levels, bounds_error=False, fill_value="extrapolate"
    )(levels)
    out = np.empty_like(pred)
    for index in range(pred.shape[0]):
        interp = scipy.interpolate.interp1d(
            levels, pred[index], kind="linear", bounds_error=False, fill_value="extrapolate"
        )
        out[index] = interp(source_levels)
    out = np.maximum.accumulate(out, axis=1)
    if domain is not None:
        low, high = domain
        if low is not None:
            out = np.maximum(out, low)
        if high is not None:
            out = np.minimum(out, high)
    return out
