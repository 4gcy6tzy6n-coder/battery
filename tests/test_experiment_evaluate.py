"""Correctness tests for the aggregation metrics that feed the paper's tables."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.integrate
import scipy.stats

from tristatelite.experiments import evaluate

DENSE_LEVELS = np.arange(0.05, 1.0, 0.05)  # 19 levels, experiment grid


# --- CRPS -------------------------------------------------------------------


def test_crps_is_zero_for_a_point_mass_forecast() -> None:
    target = np.array([1.5, 2.0, 3.0])
    pred = np.tile(target[:, None], (1, 3))  # every quantile equals the target
    assert evaluate.crps_from_quantiles(pred, target, np.array([0.05, 0.5, 0.95])) == 0.0


def test_crps_matches_exact_gaussian_within_10_percent() -> None:
    """Trapezoid CRPS from the 19-level grid must approximate the exact Gaussian CRPS."""
    mu, sigma = 0.5, 1.2
    dist = scipy.stats.norm(loc=mu, scale=sigma)
    targets = np.linspace(-2.0, 3.0, 41)
    quantiles = dist.ppf(DENSE_LEVELS)
    pred = np.tile(quantiles[None, :], (len(targets), 1))
    for y in targets:
        z = (y - mu) / sigma
        # Exact CRPS of N(mu, sigma^2) at observation y (Gneiting & Raftery 2007).
        exact = sigma * (
            z * (2.0 * scipy.stats.norm.cdf(z) - 1.0) + 2.0 * scipy.stats.norm.pdf(z) - 1.0 / np.sqrt(np.pi)
        )
        approx = evaluate.crps_from_quantiles(pred, np.full(len(targets), y), DENSE_LEVELS)
        assert approx == pytest.approx(exact, rel=0.10)


def test_crps_is_monotone_with_forecast_error() -> None:
    """A wider, miscentered forecast must score worse than a tight centered one."""
    target = np.array([0.5, 0.5, 0.5])
    tight = np.array([[0.40, 0.50, 0.60]] * 3)
    wide = np.array([[0.10, 0.50, 0.90]] * 3)
    assert evaluate.crps_from_quantiles(wide, target, np.array([0.05, 0.5, 0.95])) > evaluate.crps_from_quantiles(
        tight, target, np.array([0.05, 0.5, 0.95])
    )


# --- Pinball ----------------------------------------------------------------


def test_pinball_matches_manual_calculation() -> None:
    pred = np.array([[0.0, 0.0, 0.0]])
    target = np.array([2.0])
    levels = np.array([0.05, 0.50, 0.95])
    expected = 2.0 * (0.05 + 0.50 + 0.95) / 3.0
    assert evaluate.pinball_mean(pred, target, levels) == pytest.approx(expected)


def test_pinball_is_zero_for_exact_quantiles() -> None:
    rng = np.random.default_rng(2)
    n = 64
    target = rng.uniform(0, 1, n)
    pred = np.tile(target[:, None], (1, 3))
    assert evaluate.pinball_mean(pred, target, np.array([0.05, 0.5, 0.95])) == pytest.approx(0.0, abs=1e-9)


# --- Interval metrics -------------------------------------------------------


def test_picp_pinaw_on_synthetic_intervals() -> None:
    lower = np.array([0.0, 0.0, 0.0])
    upper = np.array([1.0, 1.0, 1.0])
    target = np.array([0.5, 0.2, 1.0])
    assert evaluate.interval_coverage(lower, upper, target) == pytest.approx(1.0)
    assert evaluate.interval_width(lower, upper) == pytest.approx(1.0)


def test_picp_counts_missed_targets() -> None:
    lower = np.array([0.0, 0.0])
    upper = np.array([0.5, 0.5])
    target = np.array([0.25, 0.75])
    assert evaluate.interval_coverage(lower, upper, target) == pytest.approx(0.5)


# --- ECE --------------------------------------------------------------------


def test_ece_zero_for_perfectly_calibrated() -> None:
    rng = np.random.default_rng(3)
    n = 20_000
    target = rng.uniform(0, 1, n)
    # Quantiles at their nominal empirical positions -> coverage matches level.
    pred = np.array([np.quantile(target, q) for q in (0.05, 0.5, 0.95)])[None, :]
    pred = np.broadcast_to(pred, (n, 3))
    assert evaluate.quantile_ece(pred, target, np.array([0.05, 0.5, 0.95])) < 0.02


def test_ece_detects_miscalibration() -> None:
    rng = np.random.default_rng(4)
    n = 10_000
    target = rng.uniform(0, 1, n)
    # Overconfident: intervals too narrow -> coverage below nominal.
    pred = np.stack([target - 0.2, target, target + 0.2], axis=1).clip(0, 1)
    ece = evaluate.quantile_ece(pred, target, np.array([0.05, 0.5, 0.95]))
    assert ece > 0.05


# --- Point metrics ----------------------------------------------------------


def test_mae_rmse_hand_computed() -> None:
    pred = np.array([1.0, 2.0, 3.0])
    target = np.array([1.0, 3.0, 3.0])
    mae, rmse, rrmse = evaluate.mae_rmse(pred, target)
    assert mae == pytest.approx(1.0 / 3.0)
    assert rmse == pytest.approx(np.sqrt(1.0 / 3.0))
    assert rrmse == pytest.approx(np.sqrt(1.0 / 3.0) / (7.0 / 3.0))


# --- Physics self-consistency -----------------------------------------------


def test_physical_consistency_error_zero_for_matching_prediction() -> None:
    soc = np.array([0.5, 0.8])
    soh = np.array([0.8, 0.9])
    q_ref = np.array([2.0, 2.0])
    current = np.array([1.0, 2.0])
    physical = soc * soh * q_ref / current * 3600.0
    log_tte = np.log1p(physical)
    error, diag = evaluate.physical_consistency_error(soc, soh, log_tte, q_ref, current)
    assert error == pytest.approx(0.0, abs=1e-9)
    assert diag["active_fraction"] == pytest.approx(1.0)


def test_physical_consistency_inactive_rows_are_excluded() -> None:
    soc = np.array([0.5, 0.5])
    soh = np.array([0.8, 0.8])
    q_ref = np.array([2.0, 2.0])
    current = np.array([1.0, 0.01])  # second row below threshold
    log_tte = np.array([1.0, 999.0])
    error, diag = evaluate.physical_consistency_error(soc, soh, log_tte, q_ref, current)
    assert diag["active_fraction"] == pytest.approx(0.5)
    assert error == pytest.approx(np.abs(1.0 - np.log1p(0.5 * 0.8 * 2.0 / 1.0 * 3600.0)))


# --- Validation -------------------------------------------------------------


def test_evaluate_state_uses_median_and_outer_levels_for_19_quantiles() -> None:
    """With 19 levels, median must be column 9 and the 90% interval the outer pair."""
    rng = np.random.default_rng(5)
    n = 20_000
    target = rng.normal(loc=0.5, scale=0.2, size=n)
    # Perfect quantile forecast: columns are the empirical quantiles of the target.
    levels = DENSE_LEVELS
    quantiles = np.quantile(target, levels)
    pred = np.tile(quantiles[None, :], (n, 1))
    bundle = evaluate.evaluate_state(pred, target, levels)
    assert bundle["picp_90"] == pytest.approx(0.90, abs=0.01)  # outer 0.05/0.95 pair
    assert bundle["ece"] < 0.01
    # Median column (index 9) equals the empirical median of the target.
    assert pred[0, 9] == pytest.approx(np.median(target), rel=0.05)


@pytest.mark.parametrize(
    "fn, kwargs",
    [
        (evaluate.pinball_mean, {}),
        (evaluate.crps_from_quantiles, {}),
        (evaluate.quantile_ece, {}),
    ],
)
def test_metrics_reject_bad_levels(fn, kwargs) -> None:
    pred = np.zeros((4, 3))
    target = np.zeros(4)
    with pytest.raises(ValueError, match="levels"):
        fn(pred, target, np.array([0.5, 0.2, 0.8]), **kwargs)
    with pytest.raises(ValueError, match="levels"):
        fn(pred, target, np.array([0.0, 0.5, 1.0]), **kwargs)


def test_metrics_reject_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="first dimension"):
        evaluate.pinball_mean(np.zeros((4, 3)), np.zeros(5), np.array([0.05, 0.5, 0.95]))
