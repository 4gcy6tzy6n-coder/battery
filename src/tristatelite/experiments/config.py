"""Frozen experiment configuration and YAML loading."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields
from pathlib import Path

import yaml

HEAD_MODES = ("ordered", "unordered", "point")


@dataclass(frozen=True)
class ExperimentConfig:
    """All hyperparameters and protocol choices for one experiment run."""

    # Data
    dataset_root: str
    split_seed: int = 2026
    fast_length: int = 128
    history_length: int = 8
    stride: int = 10
    training_pool: int = 150_000  # anchors drawn per epoch pool; <=0 uses all
    eval_pool: int = 0  # anchors used for val/test; <=0 uses all

    # Model
    encoder_type: str = "gru"  # gru | tcn
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    num_quantiles: int = 19
    head_mode: str = "ordered"

    # Training
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 60
    early_stop_patience: int = 8
    lr_patience: int = 3
    lr_schedule: str = "plateau"  # plateau | cosine
    grad_clip_norm: float = 0.0  # 0 disables gradient clipping
    warmup_epochs: int = 0
    seed: int = 0
    device: str = "cpu"

    # Loss
    physics_weight: float = 0.1
    cv_weighted: bool = True
    physics_detach_soc: bool = False
    normalize_by_target_scale: bool = True

    # Inference
    mc_samples: int = 0  # >0 enables MC-dropout averaging at evaluation

    eval_batch_size: int = 512

    def __post_init__(self) -> None:
        if self.split_seed <= 0:
            raise ValueError("split_seed must be positive")
        if self.fast_length <= 0 or self.history_length <= 0 or self.stride <= 0:
            raise ValueError("window lengths and stride must be positive")
        if self.training_pool == 0:
            raise ValueError("training_pool must be non-zero")
        if self.hidden_dim <= 0 or self.num_layers <= 0:
            raise ValueError("hidden_dim and num_layers must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.num_quantiles < 3 or self.num_quantiles % 2 == 0:
            raise ValueError("num_quantiles must be an odd integer of at least three")
        if self.head_mode not in HEAD_MODES:
            raise ValueError(f"head_mode must be one of {HEAD_MODES}")
        if self.encoder_type not in ("gru", "tcn"):
            raise ValueError("encoder_type must be 'gru' or 'tcn'")
        if self.batch_size <= 0 or self.eval_batch_size <= 0:
            raise ValueError("batch sizes must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be positive")
        if self.early_stop_patience <= 0 or self.lr_patience <= 0:
            raise ValueError("patience values must be positive")
        if self.lr_schedule not in ("plateau", "cosine"):
            raise ValueError("lr_schedule must be 'plateau' or 'cosine'")
        if self.grad_clip_norm < 0:
            raise ValueError("grad_clip_norm must be non-negative")
        if self.warmup_epochs < 0 or self.warmup_epochs >= self.max_epochs:
            raise ValueError("warmup_epochs must be in [0, max_epochs)")
        if self.physics_weight < 0:
            raise ValueError("physics_weight must be non-negative")
        if self.mc_samples < 0:
            raise ValueError("mc_samples must be non-negative")

    def config_hash(self) -> str:
        """Order-independent hash over every field except seed/device."""
        payload = json.dumps(
            {field.name: getattr(self, field.name) for field in fields(self)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()[:16]


def load_config(path: str | Path) -> ExperimentConfig:
    """Load a YAML file, mapping only known fields (unknown keys are rejected)."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("config root must be a mapping")
    known = {field.name for field in fields(ExperimentConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    return ExperimentConfig(**raw)
