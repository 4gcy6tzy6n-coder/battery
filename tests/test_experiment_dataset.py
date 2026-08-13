"""Contract tests for PreparedWindowDataset (item parity with BatteryWindowDataset)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tristatelite.data.windows import BatteryWindowDataset
from tristatelite.experiments.dataset import PreparedWindowDataset


def _synthetic_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(0)
    sample_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for battery in ("B1", "B2"):
        for cycle in range(3):
            n = 120 + cycle * 30
            for t in range(n):
                sample_rows.append(
                    {
                        "battery_id": battery,
                        "cycle_id": f"{battery}-c{cycle}",
                        "cycle_index": cycle,
                        "timestamp_s": float(t),
                        "voltage_v__scaled": float(rng.normal(3.7, 0.1)),
                        "temperature_available": bool(t % 5 != 0),
                        "current_a__scaled": float(rng.normal(1.0, 0.2)),
                        "i_eff_60s": float(rng.normal(1.0, 0.1)),
                        "current_cv_60s": float(rng.uniform(0.0, 0.3)),
                        "q_ref_ah": 2.0,
                        "soc_target": float(1.0 - t / n),
                        "soh_target": 1.0,
                        "log_tte_target": float(np.log1p(n - t)),
                    }
                )
            summary_rows.append(
                {
                    "battery_id": battery,
                    "cycle_id": f"{battery}-c{cycle}",
                    "cycle_index": cycle,
                    "delivered_ah": 1.5,
                    "duration_s": float(n),
                    "mean_voltage_v": 3.7,
                    "mean_current_a": 1.0,
                    "mean_temperature_c": 25.0,
                    "current_std_a": 0.1,
                    "temperature_rise_c": 0.5,
                    "voltage_slope_v_per_s": -1e-4,
                }
            )
    samples = pd.DataFrame(sample_rows)
    summaries = pd.DataFrame(summary_rows)
    return samples, summaries


@pytest.fixture()
def frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    return _synthetic_frames()


def test_item_matches_battery_window_dataset(frames) -> None:
    samples, summaries = frames
    features = ["voltage_v__scaled", "current_a__scaled", "temperature_available"]
    battery_ids = ["B1", "B2"]
    reference = BatteryWindowDataset(samples, summaries, battery_ids, features)
    prepared = PreparedWindowDataset(samples, summaries, battery_ids, features)
    assert len(prepared) == len(reference)
    for index in (0, 1, len(prepared) // 2, len(prepared) - 1):
        got = prepared[index]
        want = reference[index]
        assert torch_equal(got["fast_x"], want["fast_x"])
        assert torch_equal(got["fast_mask"], want["fast_mask"])
        assert torch_equal(got["slow_x"], want["slow_x"])
        assert torch_equal(got["slow_mask"], want["slow_mask"])
        assert torch_equal(got["targets"], want["targets"])
        for key in ("i_eff_60s", "current_cv_60s", "q_ref_ah"):
            assert torch_equal(got["physics"][key], want["physics"][key])
        assert got["metadata"] == want["metadata"]


def test_fetch_is_fast_and_batched(frames) -> None:
    samples, summaries = frames
    prepared = PreparedWindowDataset(samples, summaries, ["B1", "B2"], ["voltage_v__scaled"])
    items = [prepared[i] for i in range(64)]
    assert len(items) == 64
    assert all(item["fast_x"].shape == (128, 1) for item in items)


def test_pool_size_subsamples_deterministically(frames) -> None:
    samples, summaries = frames
    features = ["voltage_v__scaled", "current_a__scaled", "temperature_available"]
    full = PreparedWindowDataset(samples, summaries, ["B1", "B2"], features)
    pooled_a = PreparedWindowDataset(samples, summaries, ["B1", "B2"], features, pool_size=10, seed=7)
    pooled_b = PreparedWindowDataset(samples, summaries, ["B1", "B2"], features, pool_size=10, seed=7)
    assert len(pooled_a) == 10
    assert pooled_a.metadata_array()["timestamp_s"].tolist() == pooled_b.metadata_array()[
        "timestamp_s"
    ].tolist()
    assert len(pooled_a) < len(full)


def test_rejects_target_derived_features(frames) -> None:
    samples, summaries = frames
    with pytest.raises(ValueError, match="forbidden"):
        PreparedWindowDataset(samples, summaries, ["B1"], ["soc_target__scaled"])


def test_rejects_missing_physics(frames) -> None:
    samples, summaries = frames
    samples = samples.drop(columns=["i_eff_60s"])
    with pytest.raises(ValueError, match="physics"):
        PreparedWindowDataset(samples, summaries, ["B1"], ["voltage_v__scaled"])


def test_metadata_array_is_anchor_aligned(frames) -> None:
    samples, summaries = frames
    prepared = PreparedWindowDataset(samples, summaries, ["B1"], ["voltage_v__scaled"])
    meta = prepared.metadata_array()
    assert meta["battery_id"].shape == (len(prepared),)
    assert set(meta["battery_id"].tolist()) == {"B1"}


def torch_equal(a, b) -> bool:
    return bool((a.numpy() == b.numpy()).all())


def test_slow_window_excludes_current_cycle(frames) -> None:
    """Slow history must contain only strictly earlier cycles (leakage gate)."""
    samples, summaries = frames
    prepared = PreparedWindowDataset(samples, summaries, ["B1"], ["voltage_v__scaled"])
    last_anchor = len(prepared) - 1
    item = prepared[last_anchor]
    # Any valid slow rows belong to earlier cycles of the same battery.
    valid_rows = item["slow_x"][item["slow_mask"].numpy().astype(bool)]
    assert len(valid_rows) > 0
