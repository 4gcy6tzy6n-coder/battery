"""Safe, metadata-only ZIP archive inventory."""

import re
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

from tristatelite.provenance import sha256_file

TOKEN_SPLIT = re.compile(r"[/_.-]+")


def _unsafe_member_path(member_path: str) -> bool:
    path = PurePosixPath(member_path.replace("\\", "/"))
    return path.is_absolute() or ".." in path.parts


def inventory_zip(path: Path) -> dict[str, object]:
    """Return a JSON-serializable inventory without extracting any member."""
    path = Path(path)
    if not zipfile.is_zipfile(path):
        raise ValueError(f"not a valid ZIP archive: {path}")

    members: list[dict[str, object]] = []
    extensions: Counter[str] = Counter()
    top_level: Counter[str] = Counter()
    filename_tokens: Counter[str] = Counter()

    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            normalized_path = info.filename.replace("\\", "/")
            suffix = PurePosixPath(normalized_path).suffix.lower()
            extensions[suffix] += 1
            top_level[normalized_path.split("/", 1)[0]] += 1
            filename_tokens.update(
                token.lower() for token in TOKEN_SPLIT.split(normalized_path) if token
            )
            members.append(
                {
                    "path": normalized_path,
                    "suffix": suffix,
                    "compressed_bytes": info.compress_size,
                    "uncompressed_bytes": info.file_size,
                    "crc": info.CRC,
                    "is_encrypted": bool(info.flag_bits & 0x1),
                    "unsafe_path": _unsafe_member_path(normalized_path),
                }
            )

    members.sort(key=lambda item: str(item["path"]))
    return {
        "archive_path": str(path),
        "archive_bytes": path.stat().st_size,
        "archive_sha256": sha256_file(path),
        "member_count": len(members),
        "total_compressed_bytes": sum(int(item["compressed_bytes"]) for item in members),
        "total_uncompressed_bytes": sum(int(item["uncompressed_bytes"]) for item in members),
        "extensions": dict(sorted(extensions.items())),
        "top_level": dict(sorted(top_level.items())),
        "filename_tokens": dict(sorted(filename_tokens.items())),
        "members": members,
    }
