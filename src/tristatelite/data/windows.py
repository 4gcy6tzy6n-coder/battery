"""Backward-looking features and dual-timescale battery windows."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

SUMMARY_FEATURES = (
    "delivered_ah",
    "duration_s",
    "mean_voltage_v",
    "mean_current_a",
    "mean_temperature_c",
    "current_std_a",
    "temperature_rise_c",
    "voltage_slope_v_per_s",
)

MODEL_CONTINUOUS_FEATURES = (
    "voltage_v",
    "current_a",
    "temperature_c",
    "voltage_delta_1s",
    "current_delta_1s",
    "temperature_delta_1s",
    "current_mean_10s",
    "current_std_10s",
    "i_eff_60s",
    "voltage_slope_30s",
    "temperature_slope_60s",
)
MODEL_FEATURES = tuple(f"{name}__scaled" for name in MODEL_CONTINUOUS_FEATURES) + (
    "temperature_available",
)
PHYSICS_FEATURES = ("i_eff_60s", "current_cv_60s", "q_ref_ah")


def _causal_slope(values: pd.Series, time: pd.Series, period: int) -> pd.Series:
    elapsed = time - time.shift(period - 1)
    return ((values - values.shift(period - 1)) / elapsed).replace([np.inf, -np.inf], np.nan)


def _engineer_cycle(cycle: pd.DataFrame) -> pd.DataFrame:
    result = cycle.sort_values("timestamp_s").copy()
    result["voltage_delta_1s"] = result["voltage_v"].diff().fillna(0.0)
    result["current_delta_1s"] = result["current_a"].diff().fillna(0.0)
    result["temperature_delta_1s"] = result["temperature_c"].diff().fillna(0.0)
    result["current_mean_10s"] = result["current_a"].rolling(10, min_periods=1).mean()
    result["current_std_10s"] = (
        result["current_a"].rolling(10, min_periods=1).std(ddof=0).fillna(0.0)
    )
    result["i_eff_60s"] = result["current_a"].rolling(60, min_periods=1).mean()
    current_std_60s = result["current_a"].rolling(60, min_periods=1).std(ddof=0)
    result["current_cv_60s"] = (
        current_std_60s / (result["i_eff_60s"].abs() + 1e-3)
    ).fillna(0.0)
    result["voltage_slope_30s"] = _causal_slope(
        result["voltage_v"], result["elapsed_s"], 30
    ).fillna(0.0)
    result["temperature_slope_60s"] = _causal_slope(
        result["temperature_c"], result["elapsed_s"], 60
    ).fillna(0.0)
    return result


def engineer_causal_features(samples: pd.DataFrame) -> pd.DataFrame:
    """Add sample-count rolling features independently inside each cycle."""
    if samples.empty:
        return samples.copy()
    required = {"battery_id", "cycle_id", "timestamp_s", "elapsed_s"}
    missing = required - set(samples)
    if missing:
        raise ValueError(f"missing feature-engineering columns: {sorted(missing)}")
    frames = [
        _engineer_cycle(cycle)
        for _, cycle in samples.groupby(
            ["battery_id", "cycle_id"], sort=True, observed=True
        )
    ]
    return pd.concat(frames, ignore_index=True)


def cache_identity(
    archive_sha256: str,
    split_id: str,
    config_hash: str,
    fast_length: int,
    stride: int,
) -> str:
    """Hash every material input to a window cache."""
    payload = json.dumps(
        {
            "archive_sha256": archive_sha256,
            "split_id": split_id,
            "config_hash": config_hash,
            "fast_length": fast_length,
            "stride": stride,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


class BatteryWindowDataset(Dataset):
    """Return causal fast samples and slow completed-cycle histories."""

    def __init__(
        self,
        samples: pd.DataFrame,
        cycle_summaries: pd.DataFrame,
        battery_ids: list[str],
        feature_names: list[str],
        fast_length: int = 128,
        history_length: int = 8,
        stride: int = 10,
    ) -> None:
        if fast_length <= 0 or history_length <= 0 or stride <= 0:
            raise ValueError("window lengths and stride must be positive")
        unsafe = [name for name in feature_names if "target" in name or name == "q_ref_ah"]
        if unsafe:
            raise ValueError(f"target-derived fast features are forbidden: {unsafe}")
        missing_physics = set(PHYSICS_FEATURES) - set(samples)
        if missing_physics:
            raise ValueError(f"missing physics columns: {sorted(missing_physics)}")
        selected = set(battery_ids)
        self.samples = (
            samples[samples["battery_id"].isin(selected)]
            .sort_values(["battery_id", "cycle_index", "timestamp_s"])
            .reset_index(drop=True)
        )
        self.summaries = cycle_summaries[cycle_summaries["battery_id"].isin(selected)].copy()
        self.feature_names = list(feature_names)
        self.summary_features = [name for name in SUMMARY_FEATURES if name in self.summaries]
        self.fast_length = fast_length
        self.history_length = history_length
        self._cycle_rows: dict[tuple[str, str], np.ndarray] = {}
        self._index: list[int] = []
        for key, cycle in self.samples.groupby(
            ["battery_id", "cycle_id"], sort=True, observed=True
        ):
            rows = cycle.index.to_numpy()
            self._cycle_rows[(str(key[0]), str(key[1]))] = rows
            self._index.extend(rows[::stride].tolist())

    def __len__(self) -> int:
        return len(self._index)

    def _fast_window(self, row_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.samples.loc[row_index]
        key = (str(row["battery_id"]), str(row["cycle_id"]))
        cycle_rows = self._cycle_rows[key]
        position = int(np.flatnonzero(cycle_rows == row_index)[0])
        chosen = cycle_rows[max(0, position - self.fast_length + 1) : position + 1]
        values = self.samples.loc[chosen, self.feature_names].to_numpy(dtype=np.float32)
        padded = np.zeros((self.fast_length, len(self.feature_names)), dtype=np.float32)
        mask = np.zeros(self.fast_length, dtype=bool)
        padded[-len(values) :] = np.nan_to_num(values, nan=0.0)
        mask[-len(values) :] = True
        return torch.from_numpy(padded), torch.from_numpy(mask)

    def _slow_window(self, battery_id: str, cycle_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        prior = self.summaries[
            (self.summaries["battery_id"] == battery_id)
            & (self.summaries["cycle_index"] < cycle_index)
        ].sort_values("cycle_index")
        prior = prior.tail(self.history_length)
        values = prior[self.summary_features].to_numpy(dtype=np.float32)
        padded = np.zeros((self.history_length, len(self.summary_features)), dtype=np.float32)
        mask = np.zeros(self.history_length, dtype=bool)
        if len(values):
            padded[-len(values) :] = np.nan_to_num(values, nan=0.0)
            mask[-len(values) :] = True
        return torch.from_numpy(padded), torch.from_numpy(mask)

    def __getitem__(self, item: int) -> dict[str, object]:
        row_index = self._index[item]
        row = self.samples.loc[row_index]
        fast_x, fast_mask = self._fast_window(row_index)
        slow_x, slow_mask = self._slow_window(
            str(row["battery_id"]), int(row["cycle_index"])
        )
        targets = torch.tensor(
            [row["soc_target"], row["soh_target"], row["log_tte_target"]],
            dtype=torch.float32,
        )
        physics = {
            name: torch.tensor(row[name], dtype=torch.float32) for name in PHYSICS_FEATURES
        }
        return {
            "fast_x": fast_x,
            "fast_mask": fast_mask,
            "slow_x": slow_x,
            "slow_mask": slow_mask,
            "targets": targets,
            "physics": physics,
            "metadata": {
                "battery_id": str(row["battery_id"]),
                "cycle_id": str(row["cycle_id"]),
                "cycle_index": int(row["cycle_index"]),
                "timestamp_s": float(row["timestamp_s"]),
            },
        }
