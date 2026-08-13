"""Train-only robust feature scaling."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


def fit_scaler(train: pd.DataFrame, features: list[str]) -> dict[str, object]:
    """Fit per-feature median and IQR on an explicitly supplied training frame."""
    if train.empty:
        raise ValueError("training frame must not be empty")
    fitted: dict[str, dict[str, float]] = {}
    for feature in features:
        values = pd.to_numeric(train[feature], errors="coerce").dropna()
        if values.empty:
            raise ValueError(f"feature has no finite training values: {feature}")
        median = float(values.median())
        iqr = float(values.quantile(0.75) - values.quantile(0.25))
        fitted[feature] = {"median": median, "iqr": max(iqr, 1e-6)}
    batteries = sorted(train["battery_id"].astype(str).unique().tolist())
    return {
        "method": "median_iqr",
        "clip": [-10.0, 10.0],
        "features": fitted,
        "fitted_battery_ids": batteries,
    }


def transform_features(
    frame: pd.DataFrame,
    artifact: Mapping[str, object],
    *,
    output_suffix: str | None = None,
) -> pd.DataFrame:
    """Impute with frozen training medians, robust-scale, and clip."""
    if output_suffix is not None and not output_suffix:
        raise ValueError("output_suffix must be non-empty when provided")
    result = frame.copy()
    fitted = artifact["features"]
    if not isinstance(fitted, Mapping):
        raise TypeError("scaler features must be a mapping")
    clip = artifact.get("clip", [-10.0, 10.0])
    lower, upper = (float(value) for value in clip)
    for feature, raw_stats in fitted.items():
        source = str(feature)
        if source not in result:
            raise ValueError(f"missing source feature: {source}")
        if not isinstance(raw_stats, Mapping):
            raise TypeError(f"invalid scaler statistics for {feature}")
        median = float(raw_stats["median"])
        iqr = max(float(raw_stats["iqr"]), 1e-6)
        values = pd.to_numeric(result[source], errors="coerce").fillna(median)
        destination = source if output_suffix is None else f"{source}{output_suffix}"
        result[destination] = np.clip((values - median) / iqr, lower, upper)
    return result
