import hashlib
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError

import pytest

from tristatelite.data.download import download_file
from tristatelite.provenance import sha256_file

PAYLOAD = b"battery-data\n" * 1024


class DownloadHandler(BaseHTTPRequestHandler):
    started = threading.Event()
    release = threading.Event()
    resumable_requests = 0

    def do_GET(self):
        if self.path == "/missing":
            self.send_error(404)
            return
        if self.path == "/interrupted":
            self.send_response(200)
            self.send_header("Content-Length", str(len(PAYLOAD)))
            self.end_headers()
            self.wfile.write(PAYLOAD[:10])
            self.wfile.flush()
            self.connection.shutdown(1)
            return
        if self.path == "/resumable":
            type(self).resumable_requests += 1
            if type(self).resumable_requests == 1:
                self.send_response(200)
                self.send_header("Content-Length", str(len(PAYLOAD)))
                self.end_headers()
                self.wfile.write(PAYLOAD[:10])
                self.wfile.flush()
                self.connection.shutdown(1)
                return
            assert self.headers["Range"] == "bytes=10-"
            remainder = PAYLOAD[10:]
            self.send_response(206)
            self.send_header("Content-Length", str(len(remainder)))
            self.send_header("Content-Range", f"bytes 10-{len(PAYLOAD) - 1}/{len(PAYLOAD)}")
            self.end_headers()
            self.wfile.write(remainder)
            return

        self.send_response(200)
        self.send_header("Content-Length", str(len(PAYLOAD)))
        self.end_headers()
        if self.path == "/slow":
            type(self).started.set()
            type(self).release.wait(timeout=5)
        self.wfile.write(PAYLOAD)

    def log_message(self, format, *args):
        return


@pytest.fixture
def http_url():
    DownloadHandler.started.clear()
    DownloadHandler.release.clear()
    DownloadHandler.resumable_requests = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), DownloadHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    DownloadHandler.release.set()
    server.shutdown()
    thread.join(timeout=5)
    server.server_close()


def test_sha256_file_matches_hashlib(tmp_path: Path):
    path = tmp_path / "x.bin"
    path.write_bytes(b"battery-data")
    expected = hashlib.sha256(b"battery-data").hexdigest()
    assert sha256_file(path) == expected


def test_download_file_streams_bytes_and_returns_provenance(tmp_path: Path, http_url: str):
    destination = tmp_path / "archive.zip"

    provenance = download_file(f"{http_url}/archive", destination)

    assert destination.read_bytes() == PAYLOAD
    assert provenance["source_url"] == f"{http_url}/archive"
    assert provenance["bytes"] == len(PAYLOAD)
    assert provenance["sha256"] == hashlib.sha256(PAYLOAD).hexdigest()
    timestamp = datetime.fromisoformat(str(provenance["downloaded_at_utc"]))
    assert timestamp.utcoffset() is not None
    assert timestamp.utcoffset().total_seconds() == 0


def test_download_uses_part_file_until_transfer_completes(tmp_path: Path, http_url: str):
    destination = tmp_path / "archive.zip"

    def run_download():
        download_file(f"{http_url}/slow", destination)

    thread = threading.Thread(target=run_download)
    thread.start()
    assert DownloadHandler.started.wait(timeout=5)

    assert not destination.exists()
    part_path = destination.with_suffix(".zip.part")
    deadline = time.monotonic() + 2
    while not part_path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert part_path.exists()

    DownloadHandler.release.set()
    thread.join(timeout=5)
    assert destination.read_bytes() == PAYLOAD
    assert not part_path.exists()


def test_existing_nonempty_destination_requires_force(tmp_path: Path, http_url: str):
    destination = tmp_path / "archive.zip"
    destination.write_bytes(b"old")

    with pytest.raises(FileExistsError):
        download_file(f"{http_url}/archive", destination)

    download_file(f"{http_url}/archive", destination, force=True)
    assert destination.read_bytes() == PAYLOAD


def test_download_resumes_after_truncated_response(tmp_path: Path, http_url: str):
    destination = tmp_path / "archive.zip"

    provenance = download_file(f"{http_url}/resumable", destination)

    assert DownloadHandler.resumable_requests == 2
    assert destination.read_bytes() == PAYLOAD
    assert provenance["bytes"] == len(PAYLOAD)
    assert provenance["sha256"] == hashlib.sha256(PAYLOAD).hexdigest()


@pytest.mark.parametrize(("path", "error_type"), [("missing", HTTPError), ("interrupted", OSError)])
def test_failed_download_removes_part_file(
    tmp_path: Path, http_url: str, path: str, error_type: type[Exception]
):
    destination = tmp_path / "archive.zip"

    with pytest.raises(error_type):
        download_file(f"{http_url}/{path}", destination)

    assert not destination.exists()
    assert not destination.with_suffix(".zip.part").exists()
