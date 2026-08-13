"""Tests for empirical quantile recalibration."""

from __future__ import annotations

import numpy as np
import pytest

from tristatelite.experiments.calibration import apply_calibration, fit_calibration

LEVELS = np.arange(0.05, 1.0, 0.05)


def _coverage(pred_q: np.ndarray, targets: np.ndarray, levels: np.ndarray) -> np.ndarray:
    return np.mean(targets[:, None] <= pred_q, axis=0)


def test_calibrated_forecast_is_near_identity_map() -> None:
    rng = np.random.default_rng(0)
    n = 50_000
    target = rng.normal(0.5, 0.2, n)
    quantiles = np.quantile(target, LEVELS)
    pred = np.tile(quantiles[None, :], (n, 1))
    calibration = fit_calibration(pred, target, LEVELS)
    assert np.allclose(calibration["coverage"], LEVELS, atol=0.01)
    recal = apply_calibration(pred, LEVELS, calibration)
    assert np.allclose(recal, pred, atol=1e-3)


def test_recalibration_improves_picp_toward_nominal() -> None:
    rng = np.random.default_rng(1)
    n = 40_000
    # Systematically too-narrow intervals: quantiles shrunk 30% toward the median.
    target = rng.normal(0.5, 0.2, n)
    tight = 0.5 + 0.7 * (np.quantile(target, LEVELS) - 0.5)
    pred_tight = np.tile(tight[None, :], (n, 1))
    half = n // 2
    calibration = fit_calibration(pred_tight[:half], target[:half], LEVELS)
    raw_cov = _coverage(pred_tight[half:], target[half:], LEVELS)
    recal = apply_calibration(pred_tight[half:], LEVELS, calibration)
    recal_cov = _coverage(recal, target[half:], LEVELS)
    raw_picp = raw_cov[-1] - raw_cov[0]
    recal_picp = recal_cov[-1] - recal_cov[0]
    assert raw_picp < 0.85  # the synthetic forecast is genuinely under-dispersed
    assert recal_picp > raw_picp + 0.05  # recalibration widens toward 0.90
    assert recal_picp == pytest.approx(0.90, abs=0.08)


def test_recalibration_preserves_ordering() -> None:
    rng = np.random.default_rng(2)
    n = 20_000
    target = rng.normal(0.5, 0.3, n)
    quantiles = np.quantile(target, LEVELS)
    pred = np.tile(quantiles[None, :], (n, 1))
    calibration = fit_calibration(pred, target, LEVELS)
    recal = apply_calibration(pred, LEVELS, calibration)
    assert np.all(np.diff(recal, axis=1) >= 0)
