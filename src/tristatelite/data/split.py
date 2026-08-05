"""Deterministic physical-battery group splitting."""

from __future__ import annotations

import hashlib
import json

import numpy as np

from tristatelite.schemas import SplitManifest


def make_group_split(
    battery_ids: list[str],
    seed: int = 2026,
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
    *,
    archive_sha256: str = "",
    preprocessing_config_hash: str = "",
) -> SplitManifest:
    """Split sorted unique battery IDs with a local seeded permutation."""
    ids = sorted(set(battery_ids))
    if len(ids) != len(battery_ids):
        raise ValueError("battery_ids must be unique")
    if len(ids) < 3:
        raise ValueError("at least three batteries are required for non-empty splits")
    if len(fractions) != 3 or any(value <= 0 for value in fractions):
        raise ValueError("fractions must contain three positive values")
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("fractions must sum to one")

    shuffled = np.asarray(ids, dtype=object)[np.random.default_rng(seed).permutation(len(ids))]
    train_count = round(len(ids) * fractions[0])
    val_count = round(len(ids) * fractions[1])
    train_count = min(max(train_count, 1), len(ids) - 2)
    val_count = min(max(val_count, 1), len(ids) - train_count - 1)
    groups = (
        tuple(sorted(shuffled[:train_count].tolist())),
        tuple(sorted(shuffled[train_count : train_count + val_count].tolist())),
        tuple(sorted(shuffled[train_count + val_count :].tolist())),
    )
    identity = json.dumps(
        {"ids": ids, "seed": seed, "fractions": fractions},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return SplitManifest(
        split_id=hashlib.sha256(identity).hexdigest()[:16],
        train_batteries=groups[0],
        val_batteries=groups[1],
        test_batteries=groups[2],
        archive_sha256=archive_sha256,
        config_hash=preprocessing_config_hash,
    )
