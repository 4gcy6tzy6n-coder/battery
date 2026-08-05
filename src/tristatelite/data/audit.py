"""Executable gates for battery isolation and temporal leakage."""

from __future__ import annotations

import pandas as pd

from tristatelite.schemas import SplitManifest


def audit_leakage(
    manifest: SplitManifest,
    samples: pd.DataFrame,
    history: pd.DataFrame,
    scaler_fit_batteries: set[str],
    model_feature_names: list[str] | None = None,
) -> dict[str, object]:
    """Raise on a leakage-contract violation and return JSON-safe evidence."""
    split_sets = {
        "train": set(manifest.train_batteries),
        "val": set(manifest.val_batteries),
        "test": set(manifest.test_batteries),
    }
    overlap = (
        (split_sets["train"] & split_sets["val"])
        | (split_sets["train"] & split_sets["test"])
        | (split_sets["val"] & split_sets["test"])
    )
    if overlap:
        raise ValueError(f"battery overlap across splits: {sorted(overlap)}")

    assignment = {
        battery_id: split
        for split, battery_ids in split_sets.items()
        for battery_id in battery_ids
    }
    sample_ids = set(samples["battery_id"].astype(str))
    unassigned = sample_ids - set(assignment)
    if unassigned:
        raise ValueError(f"unassigned sample batteries: {sorted(unassigned)}")
    expected = samples["battery_id"].astype(str).map(assignment)
    if not expected.equals(samples["split"].astype(str)):
        raise ValueError("sample split does not match its battery assignment")

    if scaler_fit_batteries != split_sets["train"]:
        raise ValueError("scaler fit batteries must equal the training battery set")
    key = ["battery_id", "cycle_id", "timestamp_s"]
    if samples.duplicated(key, keep=False).any():
        raise ValueError("duplicate sample keys detected")

    if not history.empty:
        wrong_battery = set(history["battery_id"].astype(str)) - set(assignment)
        if wrong_battery:
            raise ValueError(f"unassigned history batteries: {sorted(wrong_battery)}")
        if (history["history_cycle_index"] >= history["current_cycle_index"]).any():
            raise ValueError("history must contain only strictly earlier cycles")
    forbidden = ("current_cycle_capacity", "target", "future", "final_capacity")
    unsafe_columns = [
        column for column in history.columns if any(token in column for token in forbidden)
    ]
    if unsafe_columns:
        raise ValueError(f"forbidden history feature names: {unsafe_columns}")

    model_features = model_feature_names or []
    unsafe_model_features = [
        name
        for name in model_features
        if name == "q_ref_ah" or any(token in name for token in forbidden)
    ]
    if unsafe_model_features:
        raise ValueError(f"forbidden model feature names: {unsafe_model_features}")

    return {
        "status": "passed",
        "battery_overlap_count": 0,
        "unassigned_battery_count": 0,
        "duplicate_sample_key_count": 0,
        "train_batteries": sorted(split_sets["train"]),
        "scaler_fitted_batteries": sorted(scaler_fit_batteries),
        "sample_count": len(samples),
        "history_row_count": len(history),
        "model_feature_count": len(model_features),
    }
