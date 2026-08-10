"""Aggregate per-config, per-seed results into paper-ready summary tables."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

STATE_METRICS = ("pinball", "crps", "mae", "rmse", "picp_90", "pinaw_90", "ece")
_SEED_DIR = re.compile(r"^seed(\d+)$")


def load_runs(root: Path) -> dict[str, list[dict[str, object]]]:
    """Discover runs under root/{config}/seed{seed}/results.json."""
    runs: dict[str, list[dict[str, object]]] = defaultdict(list)
    for config_dir in sorted(root.iterdir()):
        if not config_dir.is_dir():
            continue
        for seed_dir in sorted(config_dir.iterdir()):
            if not seed_dir.is_dir() or not _SEED_DIR.match(seed_dir.name):
                continue
            results_path = seed_dir / "results.json"
            if results_path.exists():
                runs[config_dir.name].append(
                    json.loads(results_path.read_text(encoding="utf-8"))
                )
    return dict(runs)


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    return float(np.mean(values)), float(np.std(values, ddof=1) if len(values) > 1 else 0.0)


def summarize(runs: dict[str, list[dict[str, object]]]) -> dict[str, object]:
    table: dict[str, object] = {}
    for config, results in runs.items():
        rows: dict[str, dict[str, object]] = {}
        for state in ("soc", "soh", "log_tte"):
            for metric in STATE_METRICS:
                values = [r["test"][state][metric] for r in results]
                mean, std = _mean_std(values)
                rows[f"{state}.{metric}"] = {"mean": mean, "std": std}
        # Aggregate TTE in seconds and physics self-consistency.
        tte_mae = [r["test"]["tte_seconds"]["mae"] for r in results]
        rows["tte_seconds.mae"] = dict(zip(("mean", "std"), _mean_std(tte_mae), strict=True))
        phys = [r["test"]["physics"]["error"] for r in results]
        rows["physics.error"] = dict(zip(("mean", "std"), _mean_std(phys), strict=True))
        epochs = [r["epochs_run"] for r in results]
        rows["epochs_run"] = {"mean": float(np.mean(epochs)), "std": float(np.std(epochs))}
        table[config] = {
            "n_seeds": len(results),
            "metrics": rows,
        }
    return table


def print_markdown(table: dict[str, object]) -> None:
    configs = list(table.keys())
    print(f"| metric | {' | '.join(configs)} |")
    print(f"| --- | {' | '.join(['---'] * len(configs))} |")
    all_metrics = list(table[configs[0]]["metrics"].keys()) if configs else []
    for metric in all_metrics:
        cells = []
        for config in configs:
            value = table[config]["metrics"][metric]
            cells.append(f"{value['mean']:.4f}±{value['std']:.4f}")
        print(f"| {metric} | {' | '.join(cells)} |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="runs directory")
    parser.add_argument("--format", choices=["json", "markdown"], default="json")
    args = parser.parse_args()
    table = summarize(load_runs(args.root))
    if args.format == "json":
        print(json.dumps(table, indent=2, sort_keys=True))
    else:
        print_markdown(table)


if __name__ == "__main__":
    main()
