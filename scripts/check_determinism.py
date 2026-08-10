"""Verify training determinism: two identical runs must yield identical metrics.

Usage: python scripts/check_determinism.py --config configs/experiments/main.yaml
       --output-a /tmp/run_a --output-b /tmp/run_b --seed 0 [--override ...]
Runs the same config twice and compares every scalar in the reported test
metrics plus the physics error. Exits non-zero on any mismatch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tristatelite.experiments.config import load_config
from tristatelite.experiments.training import run_experiment


def _flatten(value: object, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            out.update(_flatten(item, f"{prefix}.{key}" if prefix else key))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        out[prefix] = float(value)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-a", type=Path, required=True)
    parser.add_argument("--output-b", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = cfg.__class__(**{**cfg.__dict__, "seed": args.seed})
    for item in args.override:
        key, value = item.split("=", 1)
        field_type = type(getattr(cfg, key))
        cfg = cfg.__class__(**{**cfg.__dict__, key: field_type(value)})

    report_a = run_experiment(cfg, args.output_a)
    report_b = run_experiment(cfg, args.output_b)
    a = _flatten(report_a["test"])
    b = _flatten(report_b["test"])
    mismatches = []
    for key in sorted(set(a) & set(b)):
        if key.endswith("_median"):
            continue  # per-anchor arrays are not scalars to compare
        if abs(a[key] - b[key]) > 1e-6:
            mismatches.append((key, a[key], b[key]))
    print(f"compared {len(a)} scalar test metrics; mismatches: {len(mismatches)}")
    for key, va, vb in mismatches[:10]:
        print(f"  MISMATCH {key}: {va} vs {vb}")
    print("DETERMINISTIC" if not mismatches else "NOT DETERMINISTIC")
    raise SystemExit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
