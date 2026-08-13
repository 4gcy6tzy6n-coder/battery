"""NumPy-precomputed window dataset for fast CPU training.

``BatteryWindowDataset`` performs a pandas ``.loc`` lookup per item, which costs
~20 ms/item and dominates training throughput. ``PreparedWindowDataset`` runs
the identical windowing logic once and caches feature matrices plus row-index
windows, so ``__getitem__`` is pure NumPy indexing (~20 us/item). The item
structure and leakage semantics match ``BatteryWindowDataset`` exactly.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from tristatelite.data.windows import PHYSICS_FEATURES, SUMMARY_FEATURES, BatteryWindowDataset

_PAD = -1


class PreparedWindowDataset(Dataset):
    """Precomputed causal fast/slow windows with the BatteryWindowDataset item contract."""

    def __init__(
        self,
        samples: pd.DataFrame,
        cycle_summaries: pd.DataFrame,
        battery_ids: Iterable[str],
        feature_names: Iterable[str],
        fast_length: int = 128,
        history_length: int = 8,
        stride: int = 10,
        pool_size: int | None = None,
        seed: int = 0,
    ) -> None:
        if fast_length <= 0 or history_length <= 0 or stride <= 0:
            raise ValueError("window lengths and stride must be positive")
        feature_names = list(feature_names)
        unsafe = [name for name in feature_names if "target" in name or name == "q_ref_ah"]
        if unsafe:
            raise ValueError(f"target-derived fast features are forbidden: {unsafe}")
        missing_physics = set(PHYSICS_FEATURES) - set(samples)
        if missing_physics:
            raise ValueError(f"missing physics columns: {sorted(missing_physics)}")
        selected = set(battery_ids)
        # Mirror BatteryWindowDataset validation against target leakage.
        BatteryWindowDataset(
            samples,
            cycle_summaries,
            sorted(selected),
            feature_names,
            fast_length=fast_length,
            history_length=history_length,
            stride=stride,
        )

        self.feature_names = feature_names
        self.fast_length = fast_length
        self.history_length = history_length
        summary_names = [name for name in SUMMARY_FEATURES if name in cycle_summaries]

        sub = (
            samples[samples["battery_id"].astype(str).isin(selected)]
            .sort_values(["battery_id", "cycle_index", "timestamp_s"])
            .reset_index(drop=True)
        )
        summaries = cycle_summaries[
            cycle_summaries["battery_id"].astype(str).isin(selected)
        ].copy()

        self._fast_mat = sub[feature_names].to_numpy(dtype=np.float32)
        self._targets = sub[["soc_target", "soh_target", "log_tte_target"]].to_numpy(
            dtype=np.float32
        )
        self._physics = sub[list(PHYSICS_FEATURES)].to_numpy(dtype=np.float32)
        self._battery_id = sub["battery_id"].astype(str).to_numpy()
        self._cycle_id = sub["cycle_id"].astype(str).to_numpy()
        self._cycle_index = sub["cycle_index"].to_numpy(dtype=np.int64)
        self._timestamp_s = sub["timestamp_s"].to_numpy(dtype=np.float64)

        cycle_rows: dict[tuple[str, str], np.ndarray] = {}
        cycle_position = np.zeros(len(sub), dtype=np.int64)
        for key, indices in sub.groupby(
            ["battery_id", "cycle_id"], sort=True, observed=True
        ).indices.items():
            rows = np.asarray(indices, dtype=np.int64)
            cycle_rows[(str(key[0]), str(key[1]))] = rows
            cycle_position[rows] = np.arange(len(rows))

        # Match BatteryWindowDataset's per-cycle stride anchor construction exactly.
        cycle_anchor_rows: list[int] = []
        for indices in sub.groupby(
            ["battery_id", "cycle_id"], sort=True, observed=True
        ).indices.values():
            cycle_anchor_rows.extend(np.asarray(indices, dtype=np.int64)[::stride].tolist())
        anchor_rows = np.asarray(cycle_anchor_rows, dtype=np.int64)
        if pool_size is not None:
            if pool_size <= 0:
                raise ValueError("pool_size must be positive")
            if pool_size < len(anchor_rows):
                anchor_rows = np.sort(
                    np.random.default_rng(seed).choice(anchor_rows, size=pool_size, replace=False)
                )
        self._anchor_rows = anchor_rows

        windows = np.full((len(anchor_rows), fast_length), _PAD, dtype=np.int32)
        masks = np.zeros((len(anchor_rows), fast_length), dtype=bool)
        for row, anchor in enumerate(anchor_rows):
            key = (str(self._battery_id[anchor]), str(self._cycle_id[anchor]))
            rows = cycle_rows[key]
            position = int(cycle_position[anchor])
            chosen = rows[max(0, position - fast_length + 1) : position + 1]
            windows[row, -len(chosen) :] = chosen.astype(np.int32)
            masks[row, -len(chosen) :] = True
        self._windows = windows
        self._masks = masks

        slow_cache: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
        for battery_id, battery in summaries.groupby("battery_id", sort=True, observed=True):
            battery = battery.sort_values("cycle_index")
            history_matrix = battery[summary_names].to_numpy(dtype=np.float32)
            for position, cycle_index in enumerate(battery["cycle_index"].to_numpy(dtype=np.int64)):
                # Strictly earlier cycles only: the current cycle is never history.
                prior = history_matrix[max(0, position - history_length) : position]
                padded = np.zeros((history_length, len(summary_names)), dtype=np.float32)
                valid = np.zeros(history_length, dtype=bool)
                if len(prior):
                    padded[-len(prior) :] = prior
                    valid[-len(prior) :] = True
                slow_cache[(str(battery_id), int(cycle_index))] = (padded, valid)

        self._slow_cache = slow_cache
        self._summary_names = summary_names

    def __len__(self) -> int:
        return len(self._anchor_rows)

    def __getitem__(self, item: int) -> dict[str, object]:
        anchor = self._anchor_rows[item]
        fast = self._fast_mat[self._windows[item]].copy()
        fast[self._windows[item] < 0] = 0.0  # _PAD rows are zero-filled
        slow_key = (self._battery_id[anchor], int(self._cycle_index[anchor]))
        slow, slow_mask = self._slow_cache[slow_key]
        return {
            "fast_x": torch.from_numpy(fast),
            "fast_mask": torch.from_numpy(self._masks[item].copy()),
            "slow_x": torch.from_numpy(slow),
            "slow_mask": torch.from_numpy(slow_mask),
            "targets": torch.from_numpy(self._targets[anchor]),
            "physics": {
                name: torch.tensor(value, dtype=torch.float32)
                for name, value in zip(PHYSICS_FEATURES, self._physics[anchor], strict=True)
            },
            "metadata": {
                "battery_id": self._battery_id[anchor],
                "cycle_id": self._cycle_id[anchor],
                "cycle_index": int(self._cycle_index[anchor]),
                "timestamp_s": float(self._timestamp_s[anchor]),
            },
        }

    @property
    def summary_names(self) -> list[str]:
        return self._summary_names

    def metadata_array(self) -> dict[str, np.ndarray]:
        """Anchor-aligned metadata for evaluation grouping."""
        anchors = self._anchor_rows
        return {
            "battery_id": self._battery_id[anchors],
            "cycle_id": self._cycle_id[anchors],
            "cycle_index": self._cycle_index[anchors],
            "timestamp_s": self._timestamp_s[anchors],
        }
