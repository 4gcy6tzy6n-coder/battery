import zipfile
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from tristatelite.data.archive_inventory import inventory_zip
from tristatelite.data.schema_report import (
    HUMAN_DECISIONS,
    build_schema_report,
    probe_member,
    render_schema_markdown,
    validate_report_consistency,
)


def test_probe_csv_reports_structure_and_unconfirmed_role_evidence(tmp_path: Path):
    archive_path = tmp_path / "sample.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "pack01/cycle.csv",
            "Time_s,Voltage_measured,Current_A,Temperature_C,Capacity_Ah\n"
            "0,4.2,-1.0,25.0,2.1\n"
            "1,4.1,-1.0,25.1,2.1\n",
        )

    report = probe_member(archive_path, "pack01/cycle.csv")

    assert report["probe_status"] == "probed"
    assert report["encoding_candidate"] == "utf-8"
    assert report["delimiter"] == ","
    assert report["headers"] == [
        "Time_s",
        "Voltage_measured",
        "Current_A",
        "Temperature_C",
        "Capacity_Ah",
    ]
    assert report["sample_rows"][0] == ["0", "4.2", "-1.0", "25.0", "2.1"]
    assert report["row_width_consistent"] is True
    for role in ["time", "voltage", "current", "temperature", "capacity"]:
        candidates = report["candidate_roles"][role]
        assert candidates[0]["score"] > 0
        assert candidates[0]["evidence"]
        assert candidates[0]["status"] == "unconfirmed"


def test_one_character_keyword_requires_exact_header(tmp_path: Path):
    archive_path = tmp_path / "one-char.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("values.csv", "value,signal,time_value\n1,2,3\n")

    report = probe_member(archive_path, "values.csv")

    voltage_names = [item["name"] for item in report["candidate_roles"].get("voltage", [])]
    current_names = [item["name"] for item in report["candidate_roles"].get("current", [])]
    assert "value" not in voltage_names
    assert "signal" not in current_names


def test_probe_classic_mat_reports_variables_without_array_values(tmp_path: Path):
    mat_path = tmp_path / "cycle.mat"
    savemat(
        mat_path,
        {
            "time": np.arange(5, dtype=np.float64),
            "voltage": np.ones((5, 1), dtype=np.float32),
            "current": np.zeros((5, 1), dtype=np.float32),
        },
    )
    archive_path = tmp_path / "mat.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.write(mat_path, "battery01/cycle.mat")

    report = probe_member(archive_path, "battery01/cycle.mat")

    assert report["probe_status"] == "probed"
    assert report["probe_type"] == "matlab"
    variables = {item["name"]: item for item in report["mat_variables"]}
    assert variables["time"]["shape"] == [1, 5]
    assert variables["time"]["matlab_class"] == "double"
    assert variables["voltage"]["shape"] == [5, 1]
    assert "values" not in variables["current"]
    assert report["candidate_roles"]["time"][0]["status"] == "unconfirmed"


def test_build_schema_report_ranks_timeseries_and_skips_unsafe_member(tmp_path: Path):
    archive_path = tmp_path / "aggregate.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("README.txt", "Dataset documentation")
        archive.writestr("pack/cycle.csv", "time,voltage,current\n0,4.2,-1\n")
        archive.writestr("pack/notes.csv", "name,value\na,b\n")
        archive.writestr("../unsafe.csv", "time,voltage,current\n0,4.2,-1\n")
        archive.writestr("binary.bin", b"\x00\x01")

    report = build_schema_report(archive_path)

    assert report["candidate_timeseries_files"][0]["path"] == "pack/cycle.csv"
    assert report["candidate_timeseries_files"][0]["required_role_count"] == 3
    assert report["documentation_files"] == ["README.txt"]
    assert report["unsupported_extension_counts"] == {".bin": 1}
    probed_paths = {item["path"] for item in report["probes"]}
    assert "../unsafe.csv" not in probed_paths
    assert "../unsafe.csv" in report["skipped_unsafe_or_encrypted"]
    assert all(
        candidate["status"] == "unconfirmed"
        for probe in report["probes"]
        for candidates in probe.get("candidate_roles", {}).values()
        for candidate in candidates
    )


def test_markdown_report_has_required_sections_decisions_and_stop_notice(tmp_path: Path):
    archive_path = tmp_path / "report.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("README.txt", "Dataset documentation")
        archive.writestr("pack/cycle.csv", "time,voltage,current\n0,4.2,-1\n")
    inventory = inventory_zip(archive_path)
    schema = build_schema_report(archive_path)

    markdown = render_schema_markdown(inventory, schema)

    headings = [
        "## 1. Archive provenance",
        "## 2. Extension counts",
        "## 3. Top-level directory counts",
        "## 4. Candidate README/documentation files",
        "## 5. Candidate time-series files",
        "## 6. Candidate semantic fields",
        "## 7. MATLAB variable inventories",
        "## 8. Unsafe/encrypted/unsupported members",
        "## 9. Explicit human decisions required",
        "## 10. Phase 1 stop",
    ]
    assert [markdown.index(heading) for heading in headings] == sorted(
        markdown.index(heading) for heading in headings
    )
    assert all(decision in markdown for decision in HUMAN_DECISIONS)
    assert "unconfirmed" in markdown
    assert "Phase 1 stop" in markdown


def test_report_consistency_rejects_hash_mismatch_or_confirmed_candidate(tmp_path: Path):
    archive_path = tmp_path / "consistent.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("cycle.csv", "time,voltage,current\n0,4.2,-1\n")
    inventory = inventory_zip(archive_path)
    schema = build_schema_report(archive_path)
    download_manifest = {"sha256": inventory["archive_sha256"]}

    validate_report_consistency(inventory, schema, download_manifest)

    download_manifest["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="download manifest SHA256"):
        validate_report_consistency(inventory, schema, download_manifest)
    download_manifest["sha256"] = inventory["archive_sha256"]
    schema["semantic_candidates"][0]["status"] = "confirmed"
    with pytest.raises(ValueError, match="unconfirmed"):
        validate_report_consistency(inventory, schema, download_manifest)
