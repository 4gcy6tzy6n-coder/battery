"""Audited NASA CSV discovery, streaming, and discharge-run boundaries."""

import fnmatch
import zipfile
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import pandas as pd

from tristatelite.provenance import sha256_file

AUDITED_COLUMNS = [
    "start_time",
    "time",
    "mode",
    "voltage_charger",
    "temperature_battery",
    "voltage_load",
    "current_load",
    "temperature_mosfet",
    "temperature_resistor",
    "mission_type",
]
NUMERIC_COLUMNS = AUDITED_COLUMNS[1:]


@dataclass(frozen=True)
class BatteryMember:
    path: str
    battery_id: str
    group: str


def discover_battery_members(archive: Path, config: Mapping[str, object]) -> list[BatteryMember]:
    """Validate provenance and return only audited physical battery CSV members."""
    actual_hash = sha256_file(Path(archive))
    expected_hash = str(config["archive_sha256"])
    if actual_hash != expected_hash:
        raise ValueError(f"archive SHA256 mismatch: expected {expected_hash}, got {actual_hash}")

    pattern = str(config["member_pattern"])
    members = []
    with zipfile.ZipFile(archive) as zip_archive:
        for info in zip_archive.infolist():
            path = info.filename.replace("\\", "/")
            pure_path = PurePosixPath(path)
            if info.is_dir() or pure_path.is_absolute() or ".." in pure_path.parts:
                continue
            if info.flag_bits & 0x1 or not fnmatch.fnmatchcase(path, pattern):
                continue
            members.append(
                BatteryMember(
                    path=path,
                    battery_id=pure_path.stem,
                    group=pure_path.parent.name,
                )
            )
    return sorted(members, key=lambda member: member.path)


def iter_battery_chunks(
    archive: Path,
    member: BatteryMember,
    chunksize: int = 250_000,
) -> Iterator[pd.DataFrame]:
    """Stream one audited CSV and coerce documented numeric missing values."""
    if chunksize <= 0:
        raise ValueError("chunksize must be positive")
    with zipfile.ZipFile(archive) as zip_archive:
        info = zip_archive.getinfo(member.path)
        path = PurePosixPath(info.filename.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts or info.flag_bits & 0x1:
            raise ValueError(f"unsafe or encrypted member: {member.path}")
        with zip_archive.open(info) as source:
            for chunk in pd.read_csv(source, chunksize=chunksize, dtype={"start_time": "string"}):
                if list(chunk.columns) != AUDITED_COLUMNS:
                    raise ValueError(
                        f"unexpected columns in {member.path}: {list(chunk.columns)!r}"
                    )
                for column in NUMERIC_COLUMNS:
                    chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
                if chunk[["time", "mode"]].isna().any().any():
                    raise ValueError(f"missing required time/mode in {member.path}")
                modes = set(chunk["mode"].unique())
                if not modes.issubset({-1.0, 0.0, 1.0}):
                    raise ValueError(f"unsupported mode values in {member.path}: {modes}")
                yield chunk


def iter_discharge_runs(chunks: Iterable[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    """Yield maximal contiguous `mode == -1` runs, including across chunk edges."""
    buffered: list[pd.DataFrame] = []
    previous_mode: float | None = None

    for chunk in chunks:
        if chunk.empty:
            continue
        mode = chunk["mode"]
        changes = mode.ne(mode.shift())
        if previous_mode is not None:
            changes.iloc[0] = mode.iloc[0] != previous_mode
        for _, group in chunk.groupby(changes.cumsum(), sort=False):
            group_mode = float(group["mode"].iloc[0])
            if group_mode == -1.0:
                buffered.append(group.copy())
            elif buffered:
                yield pd.concat(buffered, ignore_index=True)
                buffered.clear()
            previous_mode = group_mode

    if buffered:
        yield pd.concat(buffered, ignore_index=True)
