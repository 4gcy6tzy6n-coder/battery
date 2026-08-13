"""Independently re-verify one run's reported metrics from raw predictions.

Loads ``test_predictions.npz`` and ``results.json``, recomputes every headline
metric through a *different* code path (scipy quantile-function sampling for
CRPS, direct formulas for pinball/PICP/PINAW/ECE), and reports the maximum
deviation. Any mismatch above tolerance is printed as FAILED.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.interpolate

LEVELS = np.arange(0.05, 1.0, 0.05)
_MC_TARGET_SUBSAMPLE = 2000
_MC_SAMPLES = 100000
# pinball/PICP/PINAW/ECE recompute the identical formula -> near-exact.
# CRPS uses an independent MC estimator -> allow a wider systematic gap.
TOLERANCES = {
    "pinball": 0.005,
    "picp_90": 0.005,
    "pinaw_90": 0.005,
    "ece": 0.005,
    "crps": 0.10,
}


def mc_crps(quantiles: np.ndarray, targets: np.ndarray) -> float:
    """CRPS via sampling from each anchor's piecewise-linear quantile-implied CDF.

    Independent of the trapezoid implementation: CRPS = E|X-y| - 0.5 E|X-X'|,
    with X, X' drawn per anchor from its own implied distribution.
    """
    rng = np.random.default_rng(0)
    u = rng.uniform(0.001, 0.999, size=_MC_SAMPLES)
    u2 = rng.uniform(0.001, 0.999, size=_MC_SAMPLES)
    indices = np.linspace(0, len(targets) - 1, _MC_TARGET_SUBSAMPLE).astype(int)
    values = []
    for index in indices:
        interp = scipy.interpolate.interp1d(
            LEVELS,
            quantiles[index],
            kind="linear",
            bounds_error=False,
            fill_value=(float(quantiles[index, 0]), float(quantiles[index, -1])),
        )
        x = interp(u)
        x2 = interp(u2)
        values.append(np.mean(np.abs(x - targets[index])) - 0.5 * np.mean(np.abs(x - x2)))
    return float(np.mean(values))


def pinball_direct(quantiles: np.ndarray, targets: np.ndarray) -> float:
    error = targets[:, None] - quantiles
    losses = np.maximum(LEVELS[None, :] * error, (LEVELS[None, :] - 1.0) * error)
    return float(losses.mean())


def picp_direct(quantiles: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean((targets >= quantiles[:, 0]) & (targets <= quantiles[:, -1])))


def pinaw_direct(quantiles: np.ndarray) -> float:
    return float(np.mean(quantiles[:, -1] - quantiles[:, 0]))


def ece_direct(quantiles: np.ndarray, targets: np.ndarray) -> float:
    coverage = np.mean(targets[:, None] <= quantiles, axis=0)
    return float(np.mean(np.abs(coverage - LEVELS)))


def verify_run(run_dir: Path, tolerance: float = 0.05) -> dict[str, object]:
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    npz = np.load(run_dir / "test_predictions.npz", allow_pickle=False)
    checks: dict[str, object] = {}
    max_dev = 0.0
    for state, col in (("soc", 0), ("soh", 1), ("log_tte", 2)):
        quantiles = np.asarray(npz[f"pred_{state}"], dtype=float)
        targets = np.asarray(npz["targets"], dtype=float)[:, col]
        if quantiles.shape[1] == 1:  # point baseline has no quantile metrics
            continue
        reported = results["test"][state]
        independent = {
            "pinball": pinball_direct(quantiles, targets),
            "crps": mc_crps(quantiles, targets),
            "picp_90": picp_direct(quantiles, targets),
            "pinaw_90": pinaw_direct(quantiles),
            "ece": ece_direct(quantiles, targets),
        }
        deviations = {
            key: abs(independent[key] - reported[key]) / max(abs(reported[key]), 1e-9)
            for key in independent
        }
        failed = {
            key: value for key, value in deviations.items() if value > TOLERANCES[key]
        }
        max_dev = max(max_dev, *deviations.values())
        checks[state] = {
            "reported": {key: round(float(reported[key]), 6) for key in independent},
            "independent": {key: round(float(value), 6) for key, value in independent.items()},
            "relative_deviation": {key: round(value, 4) for key, value in deviations.items()},
            "over_tolerance": {key: round(value, 4) for key, value in failed.items()},
            "quantiles_ordered": bool(np.all(np.diff(quantiles, axis=1) >= 0)),
            "finite": bool(np.isfinite(quantiles).all()),
        }
    checks["max_relative_deviation"] = round(max_dev, 4)
    over = [
        f"{state}.{metric}"
        for state in checks
        if isinstance(checks[state], dict)
        for metric in checks[state].get("over_tolerance", {})
    ]
    checks["pass"] = not over
    checks["over_tolerance_metrics"] = over
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="run directory with results.json")
    args = parser.parse_args()
    checks = verify_run(args.run)
    print(json.dumps(checks, indent=2, sort_keys=True))
    print("VERDICT:", "PASS" if checks["pass"] else "FAIL")


if __name__ == "__main__":
    main()
