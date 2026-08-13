"""Pairwise hypothesis tests across experiment configs and seeds.

Loads ``data/experiments/runs/{config}/seed{n}/`` results (and raw test
predictions where needed) and reports the H1-H4 comparisons with paired
across-seed statistics.

H1 physics value:      TST vs noPhysics  -> log_tte/soc/soh CRPS + pinball
H2 ordered heads:      TST vs unordered  -> ECE + PICP + crossing count
H3 cv weighting:       TST vs fixedW     -> TTE error by current_cv segment
H4 self-consistency:   TST vs noPhysics  -> physics.error (log1p units)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CONFIGS = (
    "main",
    "ablation_nophysics",
    "ablation_unordered",
    "ablation_fixedweight",
    "baseline_point",
    "baseline_mcdropout",
)
SEGMENTS = (0.0, 0.1, 0.3, np.inf)  # current_cv_60s segment boundaries


def _load_seed(root: Path, config: str, seed: int) -> dict[str, object]:
    return json.loads((root / config / f"seed{seed}" / "results.json").read_text(encoding="utf-8"))


def _paired(values_a: list[float], values_b: list[float]) -> dict[str, float]:
    deltas = np.asarray(values_a) - np.asarray(values_b)
    return {
        "mean_a": float(np.mean(values_a)),
        "mean_b": float(np.mean(values_b)),
        "mean_diff_a_minus_b": float(np.mean(deltas)),
        "diff_std": float(np.std(deltas, ddof=1) if len(deltas) > 1 else 0.0),
        "n": len(deltas),
    }


def _metric_values(root: Path, config: str, seeds: list[int], state: str, metric: str) -> list[float]:
    return [float(_load_seed(root, config, s)["test"][state][metric]) for s in seeds]


def _cv_stratified_tte_mae(root: Path, config: str, seed: int) -> dict[str, float]:
    """TTE MAE (seconds) within current_cv segments from raw predictions."""
    run = root / config / f"seed{seed}"
    npz = np.load(run / "test_predictions.npz", allow_pickle=False)
    current_cv = np.asarray(npz["physics"], dtype=float)[:, 1]
    targets = np.asarray(npz["targets"], dtype=float)[:, 2]
    pred = np.asarray(npz["pred_log_tte"], dtype=float)
    median_index = pred.shape[1] // 2
    tte_pred = np.expm1(pred[:, median_index])
    tte_true = np.expm1(targets)
    out: dict[str, float] = {}
    for low, high in zip(SEGMENTS[:-1], SEGMENTS[1:], strict=True):
        mask = (current_cv >= low) & (current_cv < high)
        if mask.sum() > 0:
            out[f"cv[{low:.2f},{high:.2f}]"] = float(np.mean(np.abs(tte_pred[mask] - tte_true[mask])))
        else:
            out[f"cv[{low:.2f},{high:.2f}]"] = float("nan")
    out["fraction_active"] = float(np.mean(current_cv >= 0))
    return out


def analyze(root: Path, seeds: list[int]) -> dict[str, object]:
    report: dict[str, object] = {}
    report["h1_physics"] = {
        state: _paired(
            _metric_values(root, "main", seeds, state, "crps"),
            _metric_values(root, "ablation_nophysics", seeds, state, "crps"),
        )
        for state in ("soc", "soh", "log_tte")
    }
    report["h1_physics"]["tte_seconds_mae"] = _paired(
        [float(_load_seed(root, "main", s)["test"]["tte_seconds"]["mae"]) for s in seeds],
        [float(_load_seed(root, "ablation_nophysics", s)["test"]["tte_seconds"]["mae"]) for s in seeds],
    )
    report["h2_ordered"] = {
        state: _paired(
            _metric_values(root, "main", seeds, state, "ece"),
            _metric_values(root, "ablation_unordered", seeds, state, "ece"),
        )
        for state in ("soc", "soh", "log_tte")
    }
    report["h3_cv_weighting"] = {
        f"seed{s}": _cv_stratified_tte_mae(root, "main", s) for s in seeds
    }
    report["h3_cv_weighting"]["fixed_weight"] = {
        f"seed{s}": _cv_stratified_tte_mae(root, "ablation_fixedweight", s) for s in seeds
    }
    report["h4_self_consistency"] = _paired(
        [float(_load_seed(root, "main", s)["test"]["physics"]["error"]) for s in seeds],
        [
            float(_load_seed(root, "ablation_nophysics", s)["test"]["physics"]["error"])
            for s in seeds
        ],
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/experiments/runs"))
    parser.add_argument("--seeds", type=str, default="0,1,2")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    report = analyze(args.root, seeds)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
