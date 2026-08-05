from pathlib import Path

import pytest

from tristatelite.config import config_hash, load_yaml
from tristatelite.schemas import CanonicalSample, CycleSummary, SplitManifest


def test_nasa_config_matches_audited_schema():
    config = load_yaml(Path("configs/data/nasa_randomized.yaml"))

    assert config["archive_sha256"] == (
        "89209acddbad47506d781698fc51ebe9d7cdb8968667698f63e63aa730981d21"
    )
    assert config["member_pattern"] == "battery_alt_dataset/*/battery[0-9][0-9].csv"
    assert config["fields"]["time_s"] == "time"
    assert config["fields"]["current_a"] == "current_load"
    assert config["mode_map"] == {-1: "discharge", 0: "rest", 1: "charge"}


def test_config_hash_is_order_independent():
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})
    assert len(config_hash({"a": 1})) == 64


def test_canonical_contracts_are_immutable_and_tuple_backed():
    sample = CanonicalSample(
        battery_id="battery00",
        cycle_id="battery00-discharge-00000",
        cycle_index=0,
        timestamp_s=10.0,
        elapsed_s=0.0,
        voltage_v=8.2,
        current_a=2.5,
        temperature_c=25.0,
        temperature_available=True,
        mission_type=0,
        phase="discharge",
    )
    summary = CycleSummary(
        battery_id="battery00",
        cycle_id="battery00-discharge-00000",
        cycle_index=0,
        delivered_ah=2.4,
        discharge_duration_s=3600.0,
        mean_voltage_v=7.2,
        voltage_slope_mean=-0.001,
        mean_current_a=2.5,
        current_std_a=0.01,
        mean_temperature_c=27.0,
        temperature_rise_c=4.0,
        mission_type=0,
    )
    manifest = SplitManifest(
        split_id="split-2026",
        train_batteries=("battery00",),
        val_batteries=("battery01",),
        test_batteries=("battery02",),
        archive_sha256="a" * 64,
        config_hash="b" * 64,
    )

    assert sample.phase == "discharge"
    assert summary.delivered_ah == 2.4
    assert manifest.train_batteries == ("battery00",)
    with pytest.raises(AttributeError):
        sample.phase = "rest"
