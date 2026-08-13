"""Train and evaluate one TriStateLite experiment configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tristatelite.experiments.config import load_config
from tristatelite.experiments.training import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, help="override the config seed")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="key=value overrides, e.g. --override physics_weight=0.0",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg = cfg.__class__(**{**cfg.__dict__, "seed": args.seed})
    overrides = dict(item.split("=", 1) for item in args.override)
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise ValueError(f"unknown override key: {key}")
        field_type = type(getattr(cfg, key))
        converted = field_type(value)
        cfg = cfg.__class__(**{**cfg.__dict__, key: converted})
    report = run_experiment(cfg, args.output)
    print(json.dumps({"config_hash": cfg.config_hash(), "status": "complete"}, sort_keys=True))
    print(json.dumps({"test_pinball_soc": report["test"]["soc"]["pinball"]}, sort_keys=True))


if __name__ == "__main__":
    main()
