import struct
import zipfile
from pathlib import Path

import pytest

from tristatelite.data.archive_inventory import inventory_zip


def test_inventory_zip_counts_members_extensions_and_paths(tmp_path: Path):
    path = tmp_path / "sample.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("pack01/cycle01.csv", "time,voltage,current\n0,4.2,-1\n")
        archive.writestr("README.txt", "notes")
        archive.writestr("empty/", "")

    report = inventory_zip(path)

    assert report["member_count"] == 2
    assert report["extensions"] == {".csv": 1, ".txt": 1}
    assert report["top_level"] == {"README.txt": 1, "pack01": 1}
    assert report["total_uncompressed_bytes"] > 0
    assert report["archive_bytes"] == path.stat().st_size
    assert len(report["archive_sha256"]) == 64
    assert [item["path"] for item in report["members"]] == ["README.txt", "pack01/cycle01.csv"]
    assert all(not item["unsafe_path"] for item in report["members"])
    assert report["filename_tokens"]["cycle01"] == 1
    assert report["filename_tokens"]["pack01"] == 1


@pytest.mark.parametrize("member", ["../escape.csv", "/absolute.csv", "ok/../../escape.csv"])
def test_inventory_marks_unsafe_member_paths(tmp_path: Path, member: str):
    path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(member, "x")

    report = inventory_zip(path)

    assert report["members"][0]["unsafe_path"] is True


def test_inventory_records_encrypted_flag(tmp_path: Path):
    path = tmp_path / "encrypted-flag.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("secret.csv", "time,value\n0,1\n")
    raw = bytearray(path.read_bytes())
    local = raw.index(b"PK\x03\x04")
    central = raw.index(b"PK\x01\x02")
    struct.pack_into("<H", raw, local + 6, struct.unpack_from("<H", raw, local + 6)[0] | 1)
    struct.pack_into("<H", raw, central + 8, struct.unpack_from("<H", raw, central + 8)[0] | 1)
    path.write_bytes(raw)

    report = inventory_zip(path)

    assert report["members"][0]["is_encrypted"] is True


def test_inventory_rejects_non_zip(tmp_path: Path):
    path = tmp_path / "not.zip"
    path.write_text("not a zip")

    with pytest.raises(ValueError, match="valid ZIP"):
        inventory_zip(path)
