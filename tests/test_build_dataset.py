import json
import subprocess
import zipfile
from pathlib import Path

import pandas as pd
import yaml

from tests.fixtures.synthetic_cycle import AUDITED_COLUMNS
from tristatelite.config import load_yaml
from tristatelite.provenance import sha256_file


def _battery_frame() -> pd.DataFrame:
    rows = []
    lifetime = 0.0
    for cycle_index in range(3):
        for elapsed in range(61):
            rows.append(
                {
                    "start_time": f"2022-01-{cycle_index + 1:02d} 00:00:00",
                    "time": lifetime,
                    "mode": -1.0,
                    "voltage_charger": 8.2 - elapsed * 0.04,
                    "temperature_battery": 25.0 + elapsed * 0.02,
                    "voltage_load": 8.2 - elapsed * 0.04,
                    "current_load": 1.0,
                    "temperature_mosfet": 26.0,
                    "temperature_resistor": 26.0,
                    "mission_type": 0.0,
                }
            )
            lifetime += 1.0
        rows.append(
            {
                "start_time": f"2022-01-{cycle_index + 1:02d} 00:00:00",
                "time": lifetime,
                "mode": 0.0,
                "voltage_charger": 0.0,
                "temperature_battery": 25.0,
                "voltage_load": 0.0,
                "current_load": 0.0,
                "temperature_mosfet": 26.0,
                "temperature_resistor": 26.0,
                "mission_type": 0.0,
            }
        )
        lifetime += 1.0
    return pd.DataFrame(rows)[AUDITED_COLUMNS]


def _synthetic_source(tmp_path: Path) -> tuple[Path, Path]:
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as output:
        for index in range(6):
            output.writestr(
                f"battery_alt_dataset/synthetic/battery{index:02d}.csv",
                _battery_frame().to_csv(index=False),
            )
    config = load_yaml(Path("configs/data/nasa_randomized.yaml"))
    config["archive_sha256"] = sha256_file(archive)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return archive, config_path


def test_build_cli_writes_partitioned_leakage_audited_artifacts(tmp_path: Path):
    archive, config = _synthetic_source(tmp_path)
    output = tmp_path / "processed"

    subprocess.run(
        [
            str(Path(".venv/bin/python")),
            "scripts/build_dataset.py",
            "--config",
            str(config),
            "--archive",
            str(archive),
            "--output",
            str(output),
            "--limit-batteries",
            "6",
            "--limit-cycles",
            "3",
        ],
        check=True,
    )

    expected = {
        "samples",
        "cycle_summaries.parquet",
        "split_manifest.json",
        "scaler.json",
        "rejected_cycles.csv",
        "leakage_audit.json",
        "build_report.json",
    }
    assert expected.issubset({path.name for path in output.iterdir()})
    samples = pd.read_parquet(output / "samples")
    assert not samples.duplicated(["battery_id", "cycle_id", "timestamp_s"]).any()
    assert samples["current_a"].eq(1.0).all()
    assert "current_a__scaled" in samples
    assert samples["i_eff_60s"].eq(1.0).all()
    assert samples["current_cv_60s"].eq(0.0).all()
    manifest = json.loads((output / "split_manifest.json").read_text())
    split_sets = [set(manifest[f"{name}_batteries"]) for name in ("train", "val", "test")]
    assert not split_sets[0] & split_sets[1]
    assert not split_sets[0] & split_sets[2]
    assert not split_sets[1] & split_sets[2]
    scaler = json.loads((output / "scaler.json").read_text())
    assert set(scaler["fitted_battery_ids"]) == split_sets[0]
    audit = json.loads((output / "leakage_audit.json").read_text())
    assert audit["status"] == "passed"
    report = json.loads((output / "build_report.json").read_text())
    assert report["limited"] is True
    assert report["accepted_cycle_count"] == 18
    assert set(report["physical_ranges"]) == {
        "current_a",
        "i_eff_60s",
        "current_cv_60s",
        "q_ref_ah",
    }
    assert "current_a__scaled" in report["scaled_feature_ranges"]
