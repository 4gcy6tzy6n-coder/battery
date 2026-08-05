"""Streaming downloads with atomic completion and provenance."""

import hashlib
import os
from datetime import UTC, datetime
from http.client import HTTPException, IncompleteRead
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

BLOCK_SIZE = 8 * 1024 * 1024
USER_AGENT = "TriStateLite/0.1 NASA battery data discovery"
MAX_ATTEMPTS = 5


class _IncompleteDownload(OSError):
    """A response ended before its advertised content was received."""


def download_file(url: str, destination: Path, force: bool = False) -> dict[str, object]:
    """Download *url* atomically and return JSON-serializable provenance."""
    destination = Path(destination)
    if destination.exists() and destination.stat().st_size > 0 and not force:
        raise FileExistsError(f"destination already exists: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    part_path = Path(f"{destination}.part")
    part_path.unlink(missing_ok=True)
    digest = hashlib.sha256()
    byte_count = 0
    total_bytes: int | None = None

    try:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            headers = {"User-Agent": USER_AGENT}
            if byte_count:
                headers["Range"] = f"bytes={byte_count}-"
            request = Request(url, headers=headers)

            try:
                with urlopen(request, timeout=60) as response:
                    status = getattr(response, "status", 200)
                    if byte_count and status != 206:
                        byte_count = 0
                        total_bytes = None
                        digest = hashlib.sha256()
                    mode = "ab" if status == 206 and byte_count else "wb"
                    content_length = response.headers.get("Content-Length")
                    response_bytes = 0

                    if status == 206:
                        content_range = response.headers.get("Content-Range", "")
                        expected_prefix = f"bytes {byte_count}-"
                        if not content_range.startswith(expected_prefix):
                            raise _IncompleteDownload(
                                f"invalid Content-Range for resume: {content_range!r}"
                            )
                        total_bytes = int(content_range.rsplit("/", 1)[1])
                    elif content_length is not None:
                        total_bytes = int(content_length)

                    with part_path.open(mode) as output:
                        while True:
                            try:
                                block = response.read(BLOCK_SIZE)
                            except IncompleteRead as exc:
                                block = exc.partial
                                if block:
                                    output.write(block)
                                    digest.update(block)
                                    byte_count += len(block)
                                    response_bytes += len(block)
                                raise _IncompleteDownload("response stream ended early") from exc
                            if not block:
                                break
                            output.write(block)
                            digest.update(block)
                            byte_count += len(block)
                            response_bytes += len(block)
                            print(
                                f"Downloaded {byte_count / (1024 * 1024):.2f} MiB",
                                flush=True,
                            )

                    if content_length is not None and response_bytes != int(content_length):
                        raise _IncompleteDownload(
                            f"incomplete response: expected {content_length} bytes, "
                            f"got {response_bytes}"
                        )
                    if total_bytes is not None and byte_count != total_bytes:
                        raise _IncompleteDownload(
                            f"incomplete download: expected {total_bytes} bytes, got {byte_count}"
                        )
                    break
            except HTTPError:
                raise
            except (HTTPException, OSError) as exc:
                if attempt == MAX_ATTEMPTS:
                    raise
                print(
                    f"Transfer interrupted ({exc}); retrying attempt {attempt + 1}/"
                    f"{MAX_ATTEMPTS} from byte {byte_count}",
                    flush=True,
                )
        with part_path.open("ab") as output:
            output.flush()
            os.fsync(output.fileno())
        os.replace(part_path, destination)
    except BaseException:
        part_path.unlink(missing_ok=True)
        raise

    return {
        "source_url": url,
        "path": str(destination),
        "bytes": byte_count,
        "sha256": digest.hexdigest(),
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
    }
