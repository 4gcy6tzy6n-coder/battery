from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.fixtures.synthetic_cycle import AUDITED_COLUMNS
from tristatelite.config import load_yaml
from tristatelite.data.canonicalize import canonicalize_discharge


def _config(**updates):
    config = load_yaml(Path("configs/data/nasa_randomized.yaml"))
    config.update(updates)
    return config


def _run(times=None, voltage=None, current=None):
    times = np.asarray(times if times is not None else np.arange(101), dtype=float)
    voltage = np.asarray(
        voltage if voltage is not None else np.linspace(8.2, 5.8, len(times)), dtype=float
    )
    current = np.asarray(current if current is not None else np.ones(len(times)), dtype=float)
    frame = pd.DataFrame(
        {
            "start_time": "2022-01-01 00:00:00",
            "time": times,
            "mode": -1.0,
            "voltage_charger": voltage,
            "temperature_battery": 25.0,
            "voltage_load": voltage,
            "current_load": current,
            "temperature_mosfet": 26.0,
            "temperature_resistor": 26.0,
            "mission_type": 0.0,
        }
    )
    return frame[AUDITED_COLUMNS]


def test_causal_resampling_keeps_last_observation_and_only_forward_fills():
    run = _run(times=[0.2, 0.8, 2.1, 4.2], voltage=[8.4, 8.2, 7.8, 7.2])

    canonical, audit = canonicalize_discharge(
        run,
        "battery00",
        0,
        _config(minimum_duration_s=0, minimum_samples=4),
    )

    assert audit["status"] == "accepted"
    assert canonical["timestamp_s"].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert canonical["voltage_v"].tolist() == [8.2, 8.2, 7.8, 7.8, 7.2]
    assert canonical["elapsed_s"].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert canonical["timestamp_s"].is_monotonic_increasing


@pytest.mark.parametrize(
    ("run", "config_updates", "reason"),
    [
        (_run(times=np.arange(30)), {}, "too_short"),
        (_run(times=np.linspace(0, 100, 40)), {}, "too_few_samples"),
        (_run(current=np.zeros(101)), {}, "non_positive_capacity"),
        (_run(voltage=np.linspace(6.0, 8.0, 101)), {}, "non_decreasing_voltage"),
        (_run(current=np.r_[np.ones(100), np.nan]), {}, "missing_required_signal"),
        (_run(times=np.r_[np.arange(50), np.arange(60, 111)]), {}, "gap_exceeds_limit"),
    ],
)
def test_invalid_cycles_return_specific_reason(run, config_updates, reason):
    canonical, audit = canonicalize_discharge(run, "battery00", 0, _config(**config_updates))

    assert canonical is None
    assert audit["reason"] == reason


def test_voltage_sentinel_is_ignored_for_downward_trend():
    voltage = np.linspace(8.2, 5.8, 101)
    voltage[0] = -0.026

    canonical, audit = canonicalize_discharge(_run(voltage=voltage), "battery00", 0, _config())

    assert canonical is not None
    assert audit["status"] == "accepted"
