import zipfile
from pathlib import Path

import pandas as pd
import pytest

from tests.fixtures.synthetic_cycle import AUDITED_COLUMNS, synthetic_lifetime_frame
from tristatelite.config import load_yaml
from tristatelite.data.nasa_adapter import (
    discover_battery_members,
    iter_battery_chunks,
    iter_discharge_runs,
)
from tristatelite.provenance import sha256_file


def _write_archive(path: Path) -> None:
    frame = synthetic_lifetime_frame()
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "battery_alt_dataset/regular_alt_batteries/battery00.csv", frame.to_csv(index=False)
        )
        archive.writestr("battery_alt_dataset/README.txt", "documentation")
        archive.writestr(
            "__MACOSX/battery_alt_dataset/regular_alt_batteries/._battery00.csv", "metadata"
        )


def _synthetic_config(archive: Path) -> dict[str, object]:
    config = load_yaml(Path("configs/data/nasa_randomized.yaml"))
    config["archive_sha256"] = sha256_file(archive)
    return config


def test_discovery_returns_only_sorted_physical_battery_members(tmp_path: Path):
    archive = tmp_path / "data.zip"
    _write_archive(archive)

    members = discover_battery_members(archive, _synthetic_config(archive))

    assert len(members) == 1
    assert members[0].path == "battery_alt_dataset/regular_alt_batteries/battery00.csv"
    assert members[0].battery_id == "battery00"
    assert members[0].group == "regular_alt_batteries"


def test_discovery_rejects_archive_hash_mismatch(tmp_path: Path):
    archive = tmp_path / "data.zip"
    _write_archive(archive)
    config = _synthetic_config(archive)
    config["archive_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="SHA256"):
        discover_battery_members(archive, config)


def test_streaming_coerces_documented_nan_and_preserves_columns(tmp_path: Path):
    archive = tmp_path / "data.zip"
    _write_archive(archive)
    member = discover_battery_members(archive, _synthetic_config(archive))[0]

    chunks = list(iter_battery_chunks(archive, member, chunksize=3))

    assert len(chunks) == 3
    assert list(chunks[0].columns) == AUDITED_COLUMNS
    assert pd.isna(chunks[0].loc[1, "temperature_battery"])
    assert chunks[0]["time"].dtype.kind == "f"


def test_discharge_runs_continue_across_chunks_and_ignore_start_time_change():
    frame = synthetic_lifetime_frame()
    chunks = [frame.iloc[:3].copy(), frame.iloc[3:5].copy(), frame.iloc[5:].copy()]

    runs = list(iter_discharge_runs(chunks))

    assert len(runs) == 2
    assert runs[0]["time"].tolist() == [1.0, 2.0, 3.0]
    assert runs[0]["start_time"].nunique() == 2
    assert runs[1]["time"].tolist() == [5.0, 6.0]
