"""Integration tests for the training pipeline on a minimal synthetic artifact."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from tristatelite.data.windows import MODEL_FEATURES
from tristatelite.experiments.config import ExperimentConfig, load_config
from tristatelite.experiments.training import (
    build_datasets,
    compute_loss,
    evaluate_dataset,
    run_experiment,
)
from tristatelite.losses.pinball import pinball_loss


def write_synthetic_artifact(root: Path) -> None:
    rng = np.random.default_rng(0)
    sample_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    split_assign = {"B1": "train", "B2": "val", "B3": "test"}
    for battery, split in split_assign.items():
        for cycle in range(3):
            n = 100 + cycle * 20
            for t in range(n):
                row: dict[str, object] = {
                    "battery_id": battery,
                    "cycle_id": f"{battery}-c{cycle}",
                    "cycle_index": cycle,
                    "timestamp_s": float(t),
                    "split": split,
                    "temperature_available": bool(t % 3 != 0),
                    "soc_target": float(1.0 - t / n),
                    "soh_target": float(0.9 + 0.02 * cycle),
                    "log_tte_target": float(np.log1p(n - t)),
                    "q_ref_ah": 2.0,
                    "i_eff_60s": float(rng.normal(1.0, 0.05)),
                    "current_cv_60s": float(rng.uniform(0.0, 0.2)),
                }
                for feature in MODEL_FEATURES:
                    if feature == "temperature_available":
                        row[feature] = bool(t % 3 != 0)
                    else:
                        row[feature] = float(rng.normal(0.0, 1.0))
                sample_rows.append(row)
            summary_rows.append(
                {
                    "battery_id": battery,
                    "cycle_id": f"{battery}-c{cycle}",
                    "cycle_index": cycle,
                    "delivered_ah": 1.6,
                    "duration_s": float(n),
                    "mean_voltage_v": 3.7,
                    "mean_current_a": 1.0,
                    "mean_temperature_c": 25.0,
                    "current_std_a": 0.1,
                    "temperature_rise_c": 0.4,
                    "voltage_slope_v_per_s": -1e-4,
                }
            )
    samples = pd.DataFrame(sample_rows)
    summaries = pd.DataFrame(summary_rows)
    root.mkdir(parents=True, exist_ok=True)
    samples.to_parquet(
        root / "samples", index=False, partition_cols=["split", "battery_id"], engine="pyarrow"
    )
    summaries.to_parquet(root / "cycle_summaries.parquet", index=False)
    manifest = {
        "split_id": "deadbeef",
        "train_batteries": ["B1"],
        "val_batteries": ["B2"],
        "test_batteries": ["B3"],
        "archive_sha256": "",
        "config_hash": "deadbeef",
    }
    (root / "split_manifest.json").write_text(json.dumps(manifest))


@pytest.fixture()
def artifact(tmp_path: Path) -> Path:
    write_synthetic_artifact(tmp_path)
    return tmp_path


def _tiny_cfg(root: Path) -> ExperimentConfig:
    return ExperimentConfig(
        dataset_root=str(root),
        training_pool=300,
        eval_pool=120,
        hidden_dim=8,
        num_layers=1,
        dropout=0.0,
        num_quantiles=9,
        max_epochs=5,
        early_stop_patience=3,
        batch_size=16,
        seed=0,
    )


def test_run_experiment_end_to_end(artifact: Path) -> None:
    report = run_experiment(_tiny_cfg(artifact), artifact.parent / "out")
    results_path = artifact.parent / "out" / "results.json"
    assert results_path.exists()
    assert report["test"]["anchor_count"] > 0
    for state in ("soc", "soh", "log_tte"):
        for key in ("pinball", "crps", "mae", "picp_90", "pinaw_90", "ece"):
            assert state in report["test"]
            assert key in report["test"][state]
    assert set(report["test"]["per_battery"].keys()) == {"B3"}
    assert report["test"]["physics"]["active_fraction"] > 0.0


def test_training_loss_decreases(artifact: Path) -> None:
    cfg = _tiny_cfg(artifact)
    report = run_experiment(cfg, artifact.parent / "out2")
    history = report["train_history"]
    assert len(history) >= 2
    assert history[-1]["train_loss"] < history[0]["train_loss"]


def test_head_modes_run(artifact: Path) -> None:
    for mode in ("ordered", "unordered", "point"):
        cfg = _tiny_cfg(artifact)
        cfg = cfg.__class__(**{**cfg.__dict__, "head_mode": mode})
        report = run_experiment(cfg, artifact.parent / f"out-{mode}")
        assert report["head_mode"] == mode
        if mode == "point":
            assert "crps" not in report["test"]["soc"]
        else:
            assert "crps" in report["test"]["soc"]


def test_compute_loss_shapes(artifact: Path) -> None:
    cfg = _tiny_cfg(artifact)
    train, _, _ = build_datasets(cfg)
    batch = {
        "fast_x": torch.randn(4, cfg.fast_length, len(MODEL_FEATURES)),
        "fast_mask": torch.ones(4, cfg.fast_length, dtype=torch.bool),
        "slow_x": torch.randn(4, cfg.history_length, 8),
        "slow_mask": torch.ones(4, cfg.history_length, dtype=torch.bool),
        "targets": torch.rand(4, 3),
        "physics": {
            name: torch.rand(4) + (2.0 if name == "q_ref_ah" else 0.5)
            for name in ("i_eff_60s", "current_cv_60s", "q_ref_ah")
        },
    }
    from tristatelite.experiments.training import build_net, target_scales

    model = build_net(cfg, len(MODEL_FEATURES), len(train.summary_names))
    scales = target_scales(train)
    total, components = compute_loss(model, batch, scales, cfg, torch.device("cpu"))
    assert torch.isfinite(total)
    assert "physics" in components
    assert components["physics"].item() > 0


def test_config_load_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text("dataset_root: x\nbogus_key: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown config keys"):
        load_config(path)


@pytest.mark.parametrize("bad", [{"num_quantiles": 4}, {"head_mode": "nope"}, {"physics_weight": -1}])
def test_config_rejects_invalid_values(bad: dict) -> None:
    kwargs = {"dataset_root": "x"}
    kwargs.update(bad)
    with pytest.raises(ValueError):
        ExperimentConfig(**kwargs)


def test_evaluate_dataset_matches_state_keys(artifact: Path) -> None:
    cfg = _tiny_cfg(artifact)
    train, _, test = build_datasets(cfg)
    from tristatelite.experiments.training import build_net

    model = build_net(cfg, len(MODEL_FEATURES), len(train.summary_names))
    model.eval()
    results = evaluate_dataset(model, test, cfg, torch.device("cpu"))
    assert "physics" in results and "per_battery" in results
    assert results["soc"]["pinball"] >= 0.0


def test_mc_dropout_eval_is_stable(artifact: Path) -> None:
    cfg = _tiny_cfg(artifact)
    cfg = cfg.__class__(**{**cfg.__dict__, "dropout": 0.3, "mc_samples": 5})
    report = run_experiment(cfg, artifact.parent / "out-mc")
    assert report["mc_samples"] == 5
    assert "crps" in report["test"]["log_tte"]


def test_pinball_multi_quantile_matches_manual() -> None:
    pred = torch.zeros(1, 3)
    target = torch.tensor([2.0])
    levels = torch.tensor([0.05, 0.5, 0.95])
    assert pinball_loss(pred, target, levels).item() == pytest.approx(2.0 * (0.05 + 0.5 + 0.95) / 3)
