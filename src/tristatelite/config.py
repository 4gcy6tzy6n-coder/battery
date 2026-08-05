"""Configuration loading and stable identity helpers."""

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import yaml


def load_yaml(path: Path) -> dict[str, object]:
    """Load a YAML mapping and reject non-mapping roots."""
    content = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(content, dict):
        raise TypeError(f"configuration root must be a mapping: {path}")
    return content


def config_hash(config: Mapping[str, object]) -> str:
    """Return the stable SHA256 identity of a JSON-serializable mapping."""
    payload = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
