import numpy as np
import pandas as pd

from tristatelite.data.windows import (
    BatteryWindowDataset,
    cache_identity,
    engineer_causal_features,
)


def _series(future_offset: float = 0.0) -> pd.DataFrame:
    time = np.arange(121, dtype=float)
    voltage = 8.2 - 0.01 * time
    current = 1.0 + 0.1 * np.sin(time / 10)
    temperature = 25.0 + 0.02 * time
    voltage[101:] += future_offset
    current[101:] += future_offset
    temperature[101:] += future_offset
    return pd.DataFrame(
        {
            "battery_id": "battery00",
            "cycle_id": "battery00-discharge-00000",
            "cycle_index": 0,
            "timestamp_s": time,
            "elapsed_s": time,
            "voltage_v": voltage,
            "current_a": current,
            "temperature_c": temperature,
        }
    )


def test_engineered_features_are_invariant_to_future_changes():
    first = engineer_causal_features(_series(0.0))
    second = engineer_causal_features(_series(100.0))
    engineered = [
        "voltage_delta_1s",
        "current_delta_1s",
        "current_mean_10s",
        "current_std_10s",
        "current_mean_60s",
        "current_cv_60s",
        "voltage_slope_30s",
        "temperature_slope_60s",
    ]

    pd.testing.assert_frame_equal(first.loc[:100, engineered], second.loc[:100, engineered])


def _window_frames():
    rows = []
    for cycle_index in range(2):
        for elapsed in range(5):
            rows.append(
                {
                    "battery_id": "battery00",
                    "cycle_id": f"battery00-discharge-{cycle_index:05d}",
                    "cycle_index": cycle_index,
                    "timestamp_s": float(cycle_index * 10 + elapsed),
                    "elapsed_s": float(elapsed),
                    "voltage_v": 8.0 - elapsed * 0.1,
                    "current_a": 1.0,
                    "soc_target": 1.0 - elapsed / 4,
                    "soh_target": 0.9,
                    "tte_seconds": float(4 - elapsed),
                    "log_tte_target": np.log1p(4 - elapsed),
                    "q_ref_ah": 1.0,
                }
            )
    summaries = pd.DataFrame(
        {
            "battery_id": ["battery00", "battery00"],
            "cycle_id": ["battery00-discharge-00000", "battery00-discharge-00001"],
            "cycle_index": [0, 1],
            "available_after_cycle_index": [0, 1],
            "delivered_ah": [0.9, 0.8],
            "duration_s": [4.0, 4.0],
            "mean_voltage_v": [7.8, 7.8],
            "mean_current_a": [1.0, 1.0],
            "mean_temperature_c": [25.0, 25.0],
            "current_std_a": [0.0, 0.0],
            "temperature_rise_c": [0.0, 0.0],
            "voltage_slope_v_per_s": [-0.1, -0.1],
        }
    )
    return pd.DataFrame(rows), summaries


def test_windows_left_pad_and_history_contains_only_completed_earlier_cycles():
    samples, summaries = _window_frames()
    dataset = BatteryWindowDataset(
        samples,
        summaries,
        ["battery00"],
        ["voltage_v", "current_a"],
        fast_length=4,
        history_length=2,
        stride=2,
    )

    assert len(dataset) == 6
    first = dataset[0]
    assert first["fast_mask"].tolist() == [False, False, False, True]
    assert first["fast_x"].shape == (4, 2)
    assert first["slow_mask"].tolist() == [False, False]
    cycle_one_start = dataset[3]
    assert cycle_one_start["slow_mask"].tolist() == [False, True]
    assert cycle_one_start["metadata"]["cycle_index"] == 1
    assert cycle_one_start["metadata"]["timestamp_s"] == 10.0

    evaluation = BatteryWindowDataset(
        samples,
        summaries,
        ["battery00"],
        ["voltage_v", "current_a"],
        fast_length=4,
        history_length=2,
        stride=1,
    )
    assert len(evaluation) == 10


def test_cache_identity_covers_every_material_input():
    base = cache_identity("archive", "split", "config", 128, 10)
    variants = [
        cache_identity("archive2", "split", "config", 128, 10),
        cache_identity("archive", "split2", "config", 128, 10),
        cache_identity("archive", "split", "config2", 128, 10),
        cache_identity("archive", "split", "config", 64, 10),
        cache_identity("archive", "split", "config", 128, 1),
    ]

    assert len(base) == 64
    assert all(identity != base for identity in variants)
