"""File provenance helpers."""

import hashlib
from pathlib import Path

BLOCK_SIZE = 8 * 1024 * 1024


def sha256_file(path: Path) -> str:
    """Return a lowercase SHA256 digest while reading in bounded blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(BLOCK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()
