"""Training loop, loss assembly, and evaluation orchestration."""

from __future__ import annotations

import json
import random
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tristatelite.data.windows import MODEL_FEATURES
from tristatelite.experiments import evaluate
from tristatelite.experiments.config import ExperimentConfig
from tristatelite.experiments.dataset import PreparedWindowDataset
from tristatelite.experiments.model import TriStateLiteNet
from tristatelite.losses.consistency import physics_consistency_loss
from tristatelite.losses.pinball import pinball_loss

STATE_NAMES = ("soc", "soh", "log_tte")


def quantile_levels(num_quantiles: int) -> np.ndarray:
    """Symmetric levels in (0, 1) with the median at index ``n // 2``."""
    if num_quantiles < 3 or num_quantiles % 2 == 0:
        raise ValueError("num_quantiles must be an odd integer of at least three")
    return np.linspace(0.05, 0.95, num_quantiles)


def git_revision() -> str:
    """Short HEAD revision for provenance, or 'unknown'."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except (subprocess.SubprocessError, OSError):  # pragma: no cover - environment dependent
        return "unknown"


def load_dataset_frames(cfg: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    root = Path(cfg.dataset_root)
    samples = pd.read_parquet(root / "samples")
    summaries = pd.read_parquet(root / "cycle_summaries.parquet")
    manifest = json.loads((root / "split_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("config_hash") and cfg.split_seed:
        pass  # split identity is already baked into the artifact
    batteries = {
        "train": list(manifest["train_batteries"]),
        "val": list(manifest["val_batteries"]),
        "test": list(manifest["test_batteries"]),
    }
    return samples, summaries, batteries


def build_datasets(cfg: ExperimentConfig) -> tuple[PreparedWindowDataset, ...]:
    """Return (train, val, test) prepared window datasets."""
    samples, summaries, batteries = load_dataset_frames(cfg)
    train = PreparedWindowDataset(
        samples,
        summaries,
        batteries["train"],
        list(MODEL_FEATURES),
        fast_length=cfg.fast_length,
        history_length=cfg.history_length,
        stride=cfg.stride,
        pool_size=cfg.training_pool if cfg.training_pool > 0 else None,
        seed=cfg.seed,
    )
    val = PreparedWindowDataset(
        samples,
        summaries,
        batteries["val"],
        list(MODEL_FEATURES),
        fast_length=cfg.fast_length,
        history_length=cfg.history_length,
        stride=cfg.stride,
        pool_size=cfg.eval_pool if cfg.eval_pool > 0 else None,
        seed=cfg.seed,
    )
    test = PreparedWindowDataset(
        samples,
        summaries,
        batteries["test"],
        list(MODEL_FEATURES),
        fast_length=cfg.fast_length,
        history_length=cfg.history_length,
        stride=cfg.stride,
        pool_size=cfg.eval_pool if cfg.eval_pool > 0 else None,
        seed=cfg.seed,
    )
    return train, val, test


def target_scales(dataset: PreparedWindowDataset) -> torch.Tensor:
    """Per-state mean |target| over the training pool for loss balancing."""
    values = np.stack([dataset[index]["targets"].numpy() for index in range(len(dataset))])
    return torch.tensor(np.abs(values).mean(axis=0), dtype=torch.float32)


def collate(batch: list[dict[str, object]]) -> dict[str, object]:
    """Stack a list of dataset items into a training/eval batch."""
    out: dict[str, object] = {}
    for key in ("fast_x", "fast_mask", "slow_x", "slow_mask", "targets"):
        out[key] = torch.stack([torch.as_tensor(item[key]) for item in batch])
    out["physics"] = {
        name: torch.stack([torch.as_tensor(item["physics"][name]) for item in batch])
        for name in ("i_eff_60s", "current_cv_60s", "q_ref_ah")
    }
    out["metadata"] = [item["metadata"] for item in batch]
    return out


def build_net(cfg: ExperimentConfig, fast_dim: int, slow_dim: int) -> TriStateLiteNet:
    return TriStateLiteNet(
        fast_input_dim=fast_dim,
        slow_input_dim=slow_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        num_quantiles=cfg.num_quantiles,
        head_mode=cfg.head_mode,
        encoder_type=cfg.encoder_type,
    )


def _pinball_per_state(
    pred: torch.Tensor, target: torch.Tensor, scale: float, levels: torch.Tensor
) -> torch.Tensor:
    return pinball_loss(pred, target, levels) * scale


def compute_loss(
    model: TriStateLiteNet,
    batch: dict[str, object],
    scales: torch.Tensor,
    cfg: ExperimentConfig,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return (total loss, per-component losses on the given device)."""
    soc, soh, tte = model(
        torch.as_tensor(batch["fast_x"], device=device),
        torch.as_tensor(batch["fast_mask"], device=device),
        torch.as_tensor(batch["slow_x"], device=device),
        torch.as_tensor(batch["slow_mask"], device=device),
    )
    targets = torch.as_tensor(batch["targets"], device=device)
    components: dict[str, torch.Tensor] = {}
    scale_weights = scales if cfg.normalize_by_target_scale else torch.ones_like(scales)
    if model.head_mode == "point":
        mse_soc = torch.mean((soc[:, 0] - targets[:, 0]) ** 2)
        mse_soh = torch.mean((soh[:, 0] - targets[:, 1]) ** 2)
        mse_tte = torch.mean((tte[:, 0] - targets[:, 2]) ** 2)
        components["soc"] = mse_soc * scale_weights[0]
        components["soh"] = mse_soh * scale_weights[1]
        components["tte"] = mse_tte * scale_weights[2]
        total = components["soc"] + components["soh"] + components["tte"]
        return total, components

    median_index = model.num_quantiles // 2
    levels = torch.tensor(
        quantile_levels(model.num_quantiles), dtype=torch.float32, device=device
    )
    components["soc"] = _pinball_per_state(soc, targets[:, 0], scale_weights[0], levels)
    components["soh"] = _pinball_per_state(soh, targets[:, 1], scale_weights[1], levels)
    components["tte"] = _pinball_per_state(tte, targets[:, 2], scale_weights[2], levels)
    total = components["soc"] + components["soh"] + components["tte"]

    if cfg.physics_weight > 0:
        physics = batch["physics"]
        soc_median = soc[:, median_index]
        if cfg.physics_detach_soc:
            # Avoid distorting the fast SOC state: no physics gradient into soc.
            soc_median = soc_median.detach()
        phys, _ = physics_consistency_loss(
            soc_median,
            soh[:, median_index],
            tte[:, median_index],
            torch.as_tensor(physics["q_ref_ah"], device=device),
            torch.as_tensor(physics["i_eff_60s"], device=device),
            torch.as_tensor(physics["current_cv_60s"], device=device),
            cv_decay=2.0 if cfg.cv_weighted else 0.0,
        )
        components["physics"] = phys * cfg.physics_weight
        total = total + components["physics"]
    return total, components


def _batch_generator(
    dataset: PreparedWindowDataset, batch_size: int, rng: np.random.Generator
) -> list[dict[str, object]]:
    indices = rng.permutation(len(dataset))
    for start in range(0, len(indices), batch_size):
        yield [dataset[int(i)] for i in indices[start : start + batch_size]]


@torch.no_grad()
def evaluate_dataset(
    model: TriStateLiteNet,
    dataset: PreparedWindowDataset,
    cfg: ExperimentConfig,
    device: torch.device,
    return_raw: bool = False,
) -> dict[str, object]:
    """Collect predictions over all anchors and return the full metric bundle.

    With ``return_raw=True`` the returned dict additionally carries
    ``raw_pred``, ``raw_targets``, ``raw_physics``, and ``raw_meta`` arrays so
    results can be independently re-verified and plotted.
    """
    model.eval()
    rng = np.random.default_rng(cfg.seed)
    state_pred: dict[str, list[np.ndarray]] = {name: [] for name in STATE_NAMES}
    targets_list: list[np.ndarray] = []
    physics_list: list[np.ndarray] = []
    meta: list[dict[str, object]] = []

    for batch_items in _batch_generator(dataset, cfg.eval_batch_size, rng):
        batch = collate(batch_items)
        soc, soh, tte = model(
            torch.as_tensor(batch["fast_x"], device=device),
            torch.as_tensor(batch["fast_mask"], device=device),
            torch.as_tensor(batch["slow_x"], device=device),
            torch.as_tensor(batch["slow_mask"], device=device),
        )
        preds = [soc, soh, tte]
        if cfg.mc_samples > 0:
            model.train()  # enable dropout for MC sampling
            samples: list[list[torch.Tensor]] = []
            for _ in range(cfg.mc_samples):
                mc = model(
                    torch.as_tensor(batch["fast_x"], device=device),
                    torch.as_tensor(batch["fast_mask"], device=device),
                    torch.as_tensor(batch["slow_x"], device=device),
                    torch.as_tensor(batch["slow_mask"], device=device),
                )
                samples.append(mc)
            model.eval()
            preds = [torch.stack([s[i] for s in samples], dim=0).mean(dim=0) for i in range(3)]

        for name, pred in zip(STATE_NAMES, preds, strict=True):
            state_pred[name].append(pred.cpu().numpy())
        targets_list.append(torch.as_tensor(batch["targets"]).numpy())
        physics_list.append(
            np.stack(
                [
                    torch.as_tensor(batch["physics"][key]).numpy()
                    for key in ("i_eff_60s", "current_cv_60s", "q_ref_ah")
                ],
                axis=1,
            )
        )
        meta.extend(batch["metadata"])

    targets = np.concatenate(targets_list)
    physics = np.concatenate(physics_list)
    levels = quantile_levels(model.num_quantiles)
    results: dict[str, object] = {}
    for index, name in enumerate(STATE_NAMES):
        pred = np.concatenate(state_pred[name])
        if model.head_mode == "point":
            mae, rmse, rrmse = evaluate.mae_rmse(pred[:, 0], targets[:, index])
            results[name] = {"mae": mae, "rmse": rmse, "rrmse": rrmse}
        else:
            if model.requires_quantile_sort:
                pred = np.sort(pred, axis=1)
            results[name] = evaluate.evaluate_state(pred, targets[:, index], levels)
    # TTE in seconds for interpretation (predictions are in log1p space).
    if model.head_mode != "point":
        tte_pred = np.sort(np.concatenate(state_pred["log_tte"]), axis=1) if model.requires_quantile_sort else np.concatenate(state_pred["log_tte"])
        tte_med = np.expm1(tte_pred[:, model.num_quantiles // 2])
        mae_s, rmse_s, _ = evaluate.mae_rmse(tte_med, np.expm1(targets[:, 2]))
        results["tte_seconds"] = {"mae": mae_s, "rmse": rmse_s}

    if model.head_mode != "point":
        median_index = model.num_quantiles // 2
        median_arrays: dict[str, np.ndarray] = {}
        for name in STATE_NAMES:
            pred = np.concatenate(state_pred[name])
            if model.requires_quantile_sort:
                pred = np.sort(pred, axis=1)
            median_arrays[name] = pred[:, median_index]
        phys_error, phys_diag = evaluate.physical_consistency_error(
            median_arrays["soc"],
            median_arrays["soh"],
            median_arrays["log_tte"],
            physics[:, 2],
            physics[:, 0],
        )
        results["physics"] = {"error": float(phys_error), **phys_diag}
        for name, array in median_arrays.items():
            results[f"{name}_median"] = array.tolist()

    # Per-battery breakdown for the test evaluation.
    battery_ids = [str(m["battery_id"]) for m in meta]
    per_battery: dict[str, dict[str, float]] = {}
    unique = sorted(set(battery_ids))
    for bid in unique:
        mask = np.array([b == bid for b in battery_ids])
        row: dict[str, float] = {}
        for index, name in enumerate(STATE_NAMES):
            pred = np.concatenate(state_pred[name])[mask]
            tgt = targets[mask, index]
            if model.head_mode == "point":
                mae, rmse, _ = evaluate.mae_rmse(pred[:, 0], tgt)
                row[f"{name}_mae"] = mae
                row[f"{name}_rmse"] = rmse
            else:
                if model.requires_quantile_sort:
                    pred = np.sort(pred, axis=1)
                bundle = evaluate.evaluate_state(pred, tgt, levels)
                for key in ("pinball", "crps", "mae", "picp_90", "ece"):
                    row[f"{name}_{key}"] = bundle[key]
        per_battery[bid] = row
    results["per_battery"] = per_battery
    results["anchor_count"] = len(targets)

    if return_raw:
        raw_pred: dict[str, np.ndarray] = {}
        for name in STATE_NAMES:
            pred = np.concatenate(state_pred[name])
            if model.requires_quantile_sort:
                pred = np.sort(pred, axis=1)
            raw_pred[name] = pred
        results["raw_pred"] = raw_pred
        results["raw_targets"] = targets
        results["raw_physics"] = physics
        results["raw_meta"] = {
            "battery_id": np.asarray(battery_ids, dtype=object),
            "cycle_id": np.asarray([str(m["cycle_id"]) for m in meta], dtype=object),
            "cycle_index": np.asarray([int(m["cycle_index"]) for m in meta]),
            "timestamp_s": np.asarray([float(m["timestamp_s"]) for m in meta]),
        }
    return results


def run_experiment(cfg: ExperimentConfig, out_dir: str | Path) -> dict[str, object]:
    """Train on the isolated train split, validate, and evaluate on test."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = torch.device(cfg.device)
    train_ds, val_ds, test_ds = build_datasets(cfg)
    scales = target_scales(train_ds).to(device)
    model = build_net(cfg, len(MODEL_FEATURES), len(train_ds.summary_names)).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    if cfg.lr_schedule == "cosine":
        base_lr = cfg.learning_rate * (0.1 if cfg.warmup_epochs > 0 else 1.0)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda epoch: (
                epoch / cfg.warmup_epochs if epoch < cfg.warmup_epochs else 0.5 * (1.0 + np.cos(np.pi * (epoch - cfg.warmup_epochs) / (cfg.max_epochs - cfg.warmup_epochs)))
            )
            * (base_lr / cfg.learning_rate),
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=cfg.lr_patience
        )

    rng = np.random.default_rng(cfg.seed)
    best_val = float("inf")
    best_state: dict[str, object] | None = None
    history: list[dict[str, float]] = []
    patience_left = cfg.early_stop_patience

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        for batch_items in _batch_generator(train_ds, cfg.batch_size, rng):
            batch = collate(batch_items)
            optimizer.zero_grad()
            total, _ = compute_loss(model, batch, scales, cfg, device)
            total.backward()
            if cfg.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()
            epoch_losses.append(float(total.detach()))
        train_loss = float(np.mean(epoch_losses))

        val_metrics = evaluate_dataset(model, val_ds, cfg, device)
        if model.head_mode == "point":
            val_score = float(val_metrics["log_tte"]["mae"])
            history.append({"epoch": epoch, "train_loss": train_loss, "val_log_tte_mae": val_score})
            print(
                f"epoch {epoch} train={train_loss:.4f} val_log_tte_mae={val_score:.4f}",
                flush=True,
            )
        else:
            val_score = float(val_metrics["log_tte"]["pinball"])
            history.append(
                {"epoch": epoch, "train_loss": train_loss, "val_log_tte_pinball": val_score}
            )
            print(
                f"epoch {epoch} train={train_loss:.4f} val_log_tte_pinball={val_score:.4f}",
                flush=True,
            )
        if cfg.lr_schedule == "cosine":
            scheduler.step()
        else:
            scheduler.step(val_score)
        if val_score < best_val:
            best_val = val_score
            patience_left = cfg.early_stop_patience
            best_state = {name: param.detach().clone() for name, param in model.state_dict().items()}
        else:
            patience_left -= 1
            if patience_left == 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(
        {"state_dict": best_state, "config_hash": cfg.config_hash(), "seed": cfg.seed},
        out / "best_model.pt",
    )
    test_results = evaluate_dataset(model, test_ds, cfg, device, return_raw=True)
    np.savez(
        out / "test_predictions.npz",
        **{f"pred_{name}": test_results["raw_pred"][name] for name in STATE_NAMES},
        targets=test_results["raw_targets"],
        physics=test_results["raw_physics"],
        **{f"meta_{key}": value for key, value in test_results["raw_meta"].items()},
    )
    del test_results["raw_pred"], test_results["raw_targets"], test_results["raw_physics"], test_results["raw_meta"]

    report = {
        "config": cfg.config_hash(),
        "git_revision": git_revision(),
        "seed": cfg.seed,
        "head_mode": cfg.head_mode,
        "physics_weight": cfg.physics_weight,
        "cv_weighted": cfg.cv_weighted,
        "mc_samples": cfg.mc_samples,
        "epochs_run": history[-1]["epoch"] if history else 0,
        "best_val_log_tte_pinball": best_val,
        "train_history": history,
        "test": test_results,
    }
    (out / "results.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
