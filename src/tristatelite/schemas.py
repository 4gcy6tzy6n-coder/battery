"""Immutable contracts shared by the data pipeline."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class CanonicalSample:
    battery_id: str
    cycle_id: str
    cycle_index: int
    timestamp_s: float
    elapsed_s: float
    voltage_v: float
    current_a: float
    temperature_c: float
    temperature_available: bool
    mission_type: int | None
    phase: Literal["charge", "discharge", "rest", "unknown"]


@dataclass(frozen=True)
class CycleSummary:
    battery_id: str
    cycle_id: str
    cycle_index: int
    delivered_ah: float
    discharge_duration_s: float
    mean_voltage_v: float
    voltage_slope_mean: float
    mean_current_a: float
    current_std_a: float
    mean_temperature_c: float
    temperature_rise_c: float
    mission_type: int | None


@dataclass(frozen=True)
class SplitManifest:
    split_id: str
    train_batteries: tuple[str, ...]
    val_batteries: tuple[str, ...]
    test_batteries: tuple[str, ...]
    archive_sha256: str
    config_hash: str
