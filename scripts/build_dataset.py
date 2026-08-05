"""Build leakage-safe Phase 2A artifacts from the audited NASA archive."""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from tristatelite.config import config_hash, load_yaml
from tristatelite.data.audit import audit_leakage
from tristatelite.data.canonicalize import canonicalize_discharge
from tristatelite.data.labels import label_battery_cycles
from tristatelite.data.nasa_adapter import (
    discover_battery_members,
    iter_battery_chunks,
    iter_discharge_runs,
)
from tristatelite.data.scaling import fit_scaler, transform_features
from tristatelite.data.split import make_group_split
from tristatelite.data.windows import SUMMARY_FEATURES, engineer_causal_features

SCALED_FEATURES = ["voltage_v", "current_a", "temperature_c"]
REJECTION_COLUMNS = [
    "battery_id",
    "cycle_index",
    "status",
    "reason",
    "raw_samples",
    "duration_s",
    "delivered_ah",
]


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _history_links(summaries: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for battery_id, battery in summaries.groupby("battery_id", sort=True, observed=True):
        battery = battery.sort_values("cycle_index")
        for current_index in battery["cycle_index"]:
            for _, prior in battery[battery["cycle_index"] < current_index].iterrows():
                row = {
                    "battery_id": battery_id,
                    "current_cycle_index": int(current_index),
                    "history_cycle_index": int(prior["cycle_index"]),
                }
                row.update({name: prior[name] for name in SUMMARY_FEATURES if name in prior})
                rows.append(row)
    columns = ["battery_id", "current_cycle_index", "history_cycle_index", *SUMMARY_FEATURES]
    return pd.DataFrame(rows, columns=columns)


def build_dataset(
    config_path: Path,
    archive: Path,
    output: Path,
    *,
    split_seed: int = 2026,
    protocol: str = "calibrated",
    limit_batteries: int | None = None,
    limit_cycles: int | None = None,
) -> dict[str, object]:
    """Run the audited build and atomically publish the resulting directory."""
    config = load_yaml(config_path)
    members = discover_battery_members(archive, config)
    if limit_batteries is not None:
        if limit_batteries <= 0:
            raise ValueError("limit_batteries must be positive")
        members = members[:limit_batteries]
    if len(members) < 3:
        raise ValueError("at least three batteries are required to build isolated splits")
    if limit_cycles is not None and limit_cycles <= 0:
        raise ValueError("limit_cycles must be positive")

    all_cycles: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    rejected: list[dict[str, object]] = []
    for member in members:
        accepted: list[pd.DataFrame] = []
        runs = iter_discharge_runs(iter_battery_chunks(archive, member))
        for run_index, run in enumerate(runs):
            if limit_cycles is not None and run_index >= limit_cycles:
                break
            canonical, audit = canonicalize_discharge(
                run, member.battery_id, run_index, config
            )
            if canonical is None:
                rejected.append(audit)
            else:
                accepted.append(canonical)
        labeled, summaries = label_battery_cycles(accepted, protocol=protocol)
        summaries["source_group"] = member.group
        all_cycles.extend(labeled)
        summary_frames.append(summaries)

    samples = pd.concat(all_cycles, ignore_index=True)
    summaries = pd.concat(summary_frames, ignore_index=True)
    preprocessing_hash = config_hash(config)
    manifest = make_group_split(
        [member.battery_id for member in members],
        seed=split_seed,
        archive_sha256=str(config["archive_sha256"]),
        preprocessing_config_hash=preprocessing_hash,
    )
    assignment = {
        battery_id: split
        for split, battery_ids in (
            ("train", manifest.train_batteries),
            ("val", manifest.val_batteries),
            ("test", manifest.test_batteries),
        )
        for battery_id in battery_ids
    }
    samples["split"] = samples["battery_id"].map(assignment)
    summaries["split"] = summaries["battery_id"].map(assignment)
    scaler = fit_scaler(samples[samples["split"] == "train"], SCALED_FEATURES)
    samples = transform_features(samples, scaler)
    samples = engineer_causal_features(samples)
    history = _history_links(summaries)
    audit = audit_leakage(
        manifest,
        samples,
        history,
        set(scaler["fitted_battery_ids"]),
    )
    report = {
        "status": "complete",
        "limited": limit_batteries is not None or limit_cycles is not None,
        "archive_sha256": str(config["archive_sha256"]),
        "config_hash": preprocessing_hash,
        "split_id": manifest.split_id,
        "protocol": protocol,
        "source_battery_count": len(members),
        "accepted_cycle_count": len(summaries),
        "rejected_cycle_count": len(rejected),
        "sample_count": len(samples),
        "limit_batteries": limit_batteries,
        "limit_cycles": limit_cycles,
    }

    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary)
        samples.to_parquet(
            staging / "samples",
            index=False,
            partition_cols=["split", "battery_id"],
            engine="pyarrow",
        )
        summaries.to_parquet(staging / "cycle_summaries.parquet", index=False)
        _write_json(staging / "split_manifest.json", asdict(manifest))
        _write_json(staging / "scaler.json", scaler)
        pd.DataFrame(rejected, columns=REJECTION_COLUMNS).to_csv(
            staging / "rejected_cycles.csv", index=False
        )
        _write_json(staging / "leakage_audit.json", audit)
        _write_json(staging / "build_report.json", report)
        staging.rename(output)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--protocol", choices=["calibrated"], default="calibrated")
    parser.add_argument("--limit-batteries", type=int)
    parser.add_argument("--limit-cycles", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_dataset(
        args.config,
        args.archive,
        args.output,
        split_seed=args.split_seed,
        protocol=args.protocol,
        limit_batteries=args.limit_batteries,
        limit_cycles=args.limit_cycles,
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
