import json
import subprocess
import sys
import zipfile
from pathlib import Path

from tristatelite.provenance import sha256_file


def test_inventory_cli_writes_consistent_json_and_markdown(tmp_path: Path):
    archive_path = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("pack/cycle.csv", "time,voltage,current\n0,4.2,-1\n")
    download_manifest = tmp_path / "download.json"
    download_manifest.write_text(json.dumps({"sha256": sha256_file(archive_path)}))
    inventory_json = tmp_path / "inventory.json"
    schema_json = tmp_path / "schema.json"
    schema_markdown = tmp_path / "schema.md"
    repository_root = Path(__file__).parents[1]

    result = subprocess.run(
        [
            sys.executable,
            "scripts/inventory_nasa.py",
            "--archive",
            str(archive_path),
            "--download-manifest",
            str(download_manifest),
            "--inventory-json",
            str(inventory_json),
            "--schema-json",
            str(schema_json),
            "--schema-markdown",
            str(schema_markdown),
        ],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    inventory = json.loads(inventory_json.read_text())
    schema = json.loads(schema_json.read_text())
    assert inventory["archive_sha256"] == schema["archive_sha256"]
    assert schema["candidate_timeseries_files"][0]["path"] == "pack/cycle.csv"
    assert "Phase 1 stop" in schema_markdown.read_text()
