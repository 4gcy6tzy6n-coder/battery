"""Generate paper figures from experiment runs' saved test predictions.

Figures:
1. Calibration reliability: empirical vs nominal quantile coverage per state.
2. Physics self-consistency: predicted log-TTE median vs the physical
   ``log1p(soc*soh*q_ref/i_eff*3600)`` relationship on held-out test anchors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

LEVELS = np.arange(0.05, 1.0, 0.05)
STATE_NAMES = ("soc", "soh", "log_tte")
MEDIAN_INDEX = len(LEVELS) // 2


def load_run(run_dir: Path) -> tuple[dict[str, object], np.lib.npyio.NpzFile]:
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    npz = np.load(run_dir / "test_predictions.npz", allow_pickle=False)
    return results, npz


def calibration_axis(ax, npz, state: str) -> None:
    quantiles = np.asarray(npz[f"pred_{state}"], dtype=float)
    if quantiles.shape[1] == 1:
        ax.set_title(f"{state} (point)")
        return
    column = {"soc": 0, "soh": 1, "log_tte": 2}[state]
    targets = np.asarray(npz["targets"], dtype=float)[:, column]
    coverage = np.mean(targets[:, None] <= quantiles, axis=0)
    ax.plot(LEVELS, LEVELS, "k--", lw=1, label="ideal")
    ax.plot(LEVELS, coverage, marker="o", ms=3, lw=1, label="empirical")
    ax.set_xlabel("nominal quantile")
    ax.set_ylabel("empirical coverage")
    ax.set_title(f"{state} calibration (ECE={np.mean(np.abs(coverage - LEVELS)):.3f})")
    ax.legend(fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def physics_axis(ax, npz) -> None:
    targets = np.asarray(npz["targets"], dtype=float)
    physics = np.asarray(npz["physics"], dtype=float)
    median_index = MEDIAN_INDEX
    predicted = np.asarray(npz["pred_log_tte"], dtype=float)[:, median_index]
    soc = np.asarray(npz["pred_soc"], dtype=float)[:, median_index]
    soh = np.asarray(npz["pred_soh"], dtype=float)[:, median_index]
    q_ref = physics[:, 2]
    current = physics[:, 0]
    active = current > 0.05
    physical = soc * soh * q_ref / np.maximum(current, 1e-6) * 3600.0
    physical_log = np.log1p(np.clip(physical, 0.0, None))
    ax.scatter(physical_log[active], predicted[active], s=1, alpha=0.3, rasterized=True)
    limit = np.percentile(np.concatenate([physical_log[active], predicted[active]]), 99)
    ax.plot([0, limit], [0, limit], "r--", lw=1)
    ax.set_xlabel("physical log-TTE (log1p(soc*soh*q_ref/i_eff*3600))")
    ax.set_ylabel("predicted log-TTE median")
    ax.set_title("physics self-consistency")
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=str, help="comma-separated name=run_dir pairs")
    parser.add_argument("--output", type=Path, required=True, help="figure output directory")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    pairs = [item.split("=", 1) for item in args.runs.split(",")]

    fig, axes = plt.subplots(len(pairs), 3, figsize=(12, 3.2 * len(pairs)), squeeze=False)
    for row, (name, run_dir) in enumerate(pairs):
        _, npz = load_run(Path(run_dir))
        for col, state in enumerate(STATE_NAMES):
            calibration_axis(axes[row, col], npz, state)
        axes[row, 0].set_ylabel(f"{name}\nempirical coverage")
    fig.tight_layout()
    fig.savefig(args.output / "calibration.png", dpi=150, bbox_inches="tight")

    fig2, axes2 = plt.subplots(1, len(pairs), figsize=(4.2 * len(pairs), 4), squeeze=False)
    for col, (name, run_dir) in enumerate(pairs):
        physics_axis(axes2[0, col], np.load(Path(run_dir) / "test_predictions.npz", allow_pickle=False))
        axes2[0, col].set_title(f"{name} physics self-consistency")
    fig2.tight_layout()
    fig2.savefig(args.output / "physics_consistency.png", dpi=150, bbox_inches="tight")
    print("figures written to", args.output)


if __name__ == "__main__":
    main()
