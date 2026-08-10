"""Post-hoc empirical recalibration for a trained run's test metrics.

Loads ``best_model.pt`` + config from a run directory, evaluates on the val
split to fit the per-state coverage map, then applies it to the test split and
recomputes every probabilistic metric. Writes ``recalibrated_results.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tristatelite.data.windows import MODEL_FEATURES
from tristatelite.experiments import evaluate
from tristatelite.experiments.calibration import apply_calibration, fit_calibration
from tristatelite.experiments.config import load_config
from tristatelite.experiments.dataset import PreparedWindowDataset
from tristatelite.experiments.training import STATE_NAMES, build_net, collate, quantile_levels

LEVELS = quantile_levels(19)


@torch.no_grad()
def _collect(
    model: torch.nn.Module, dataset: PreparedWindowDataset, cfg, device: torch.device
) -> dict[str, np.ndarray]:
    model.eval()
    pred: dict[str, list[np.ndarray]] = {name: [] for name in STATE_NAMES}
    targets_list: list[np.ndarray] = []
    for start in range(0, len(dataset), cfg.eval_batch_size):
        batch = collate([dataset[i] for i in range(start, min(start + cfg.eval_batch_size, len(dataset)))])
        soc, soh, tte = model(
            torch.as_tensor(batch["fast_x"], device=device),
            torch.as_tensor(batch["fast_mask"], device=device),
            torch.as_tensor(batch["slow_x"], device=device),
            torch.as_tensor(batch["slow_mask"], device=device),
        )
        for name, out in zip(STATE_NAMES, (soc, soh, tte), strict=True):
            pred[name].append(out.cpu().numpy())
        targets_list.append(np.asarray(batch["targets"]))
    return {
        **{name: np.concatenate(v) for name, v in pred.items()},
        "targets": np.concatenate(targets_list),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg.device)
    checkpoint = torch.load(args.run / "best_model.pt", map_location="cpu")
    cfg = cfg.__class__(**{**cfg.__dict__, "seed": checkpoint["seed"]})
    from tristatelite.experiments.training import build_datasets

    train, val, test = build_datasets(cfg)
    model = build_net(cfg, len(MODEL_FEATURES), len(train.summary_names)).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    val_pred = _collect(model, val, cfg, device)
    test_pred = _collect(model, test, cfg, device)

    domains = {"soc": (0.0, 1.0), "soh": (0.0, 1.0), "log_tte": (0.0, None)}
    out: dict[str, object] = {"seed": cfg.seed}
    for state, col in (("soc", 0), ("soh", 1), ("log_tte", 2)):
        q = test_pred[state]
        target = test_pred["targets"][:, col]
        if q.shape[1] == 1:  # point baseline
            continue
        calibration = fit_calibration(val_pred[state], val_pred["targets"][:, col], LEVELS)
        q_recal = apply_calibration(q, LEVELS, calibration, domain=domains[state])
        raw = evaluate.evaluate_state(q, target, LEVELS)
        recal = evaluate.evaluate_state(q_recal, target, LEVELS)
        out[state] = {
            "raw": raw,
            "recalibrated": recal,
            "calibration_coverage": calibration["coverage"].tolist(),
        }
    (args.run / "recalibrated_results.json").write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(out, sort_keys=True))


if __name__ == "__main__":
    main()
