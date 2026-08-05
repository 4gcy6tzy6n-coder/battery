"""Causal normalization of audited NASA randomized-discharge runs."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


def _rejected(reason: str, **details: object) -> tuple[None, dict[str, object]]:
    return None, {"status": "rejected", "reason": reason, **details}


def canonicalize_discharge(
    run: pd.DataFrame,
    battery_id: str,
    cycle_index: int,
    config: Mapping[str, object],
) -> tuple[pd.DataFrame | None, dict[str, object]]:
    """Validate and causally resample one maximal discharge run to 1 Hz.

    Each one-second bin keeps its final observed sample. Empty bins only inherit
    past values, bounded by ``max_forward_fill_s``; no backward fill or
    interpolation is used.
    """
    raw_samples = len(run)
    audit_base = {
        "battery_id": battery_id,
        "cycle_index": cycle_index,
        "raw_samples": raw_samples,
    }
    if raw_samples == 0:
        return _rejected("too_short", **audit_base)

    time_s = pd.to_numeric(run["time"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(time_s).all():
        return _rejected("missing_required_signal", **audit_base)
    if np.any(np.diff(time_s) <= 0):
        return _rejected("non_increasing_time", **audit_base)

    duration_s = float(time_s[-1] - time_s[0])
    audit_base["duration_s"] = duration_s
    if duration_s < float(config["minimum_duration_s"]):
        return _rejected("too_short", **audit_base)
    if raw_samples < int(config["minimum_samples"]):
        return _rejected("too_few_samples", **audit_base)

    voltage = pd.to_numeric(run["voltage_load"], errors="coerce").to_numpy(dtype=float)
    current = pd.to_numeric(run["current_load"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(voltage).all() or not np.isfinite(current).all():
        return _rejected("missing_required_signal", **audit_base)

    delivered_ah = float(np.trapezoid(current, time_s) / 3600.0)
    audit_base["delivered_ah"] = delivered_ah
    if delivered_ah <= 0:
        return _rejected("non_positive_capacity", **audit_base)

    valid_voltage = voltage[voltage > float(config["minimum_valid_voltage_v"])]
    if len(valid_voltage) < 2 or valid_voltage[0] <= valid_voltage[-1]:
        return _rejected("non_decreasing_voltage", **audit_base)

    working = run.copy()
    working["timestamp_s"] = np.floor(time_s).astype(np.int64)
    observed = working.drop_duplicates("timestamp_s", keep="last").set_index("timestamp_s")
    full_index = pd.RangeIndex(int(observed.index[0]), int(observed.index[-1]) + 1)
    canonical = observed.reindex(full_index)

    temperature_available = canonical["temperature_battery"].notna()
    fill_columns = [
        "voltage_load",
        "current_load",
        "temperature_battery",
        "mission_type",
    ]
    canonical[fill_columns] = canonical[fill_columns].ffill(
        limit=int(config["max_forward_fill_s"])
    )
    if canonical[["voltage_load", "current_load"]].isna().any(axis=None):
        return _rejected("gap_exceeds_limit", **audit_base)

    timestamps = canonical.index.to_numpy(dtype=float)
    cycle_id = f"{battery_id}-discharge-{cycle_index:05d}"
    result = pd.DataFrame(
        {
            "battery_id": battery_id,
            "cycle_id": cycle_id,
            "cycle_index": cycle_index,
            "timestamp_s": timestamps,
            "elapsed_s": timestamps - timestamps[0],
            "voltage_v": canonical["voltage_load"].to_numpy(dtype=float),
            "current_a": canonical["current_load"].to_numpy(dtype=float),
            "temperature_c": pd.to_numeric(
                canonical["temperature_battery"], errors="coerce"
            ).to_numpy(dtype=float),
            "temperature_available": temperature_available.to_numpy(dtype=bool),
            "mission_type": pd.to_numeric(canonical["mission_type"], errors="coerce"),
            "phase": "discharge",
        }
    )
    audit = {
        "status": "accepted",
        **audit_base,
        "cycle_id": cycle_id,
        "canonical_samples": len(result),
    }
    return result, audit
