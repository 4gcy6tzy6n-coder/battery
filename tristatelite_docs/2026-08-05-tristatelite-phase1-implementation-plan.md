# TriStateLite Phase 1 Data Discovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a reproducible Python repository, download the official NASA Randomized and Recommissioned Battery archive, and generate a machine-readable and human-readable schema report without inventing any data fields.

**Architecture:** Phase 1 contains no model training and no semantic battery parsing. A streaming downloader records provenance; an archive inspector inventories members and probes supported tabular/MAT files; a report generator identifies candidate battery, cycle, time, voltage, current, temperature, phase, and capacity fields for human confirmation.

**Tech Stack:** Python 3.11, NumPy, pandas, SciPy, PyYAML, PyArrow, PyTorch, pytest, ruff.

## Global Constraints

- Use the official NASA resource URL `https://data.nasa.gov/docs/legacy/battery_alt_dataset.zip`.
- Never guess the archive’s directory structure, battery IDs, cycle IDs, or column names.
- Do not implement labels, splits, windows, models, or training in Phase 1.
- Download through a temporary `.part` file and atomically rename after success.
- Record SHA256, file size, HTTP source URL, and UTC download time.
- Inspect archive members without executing macros, scripts, notebooks, or binaries.
- Extract only individual files into a temporary directory when a parser requires a real path.
- Every command must be reproducible from the repository root.

---

## File Structure

```text
tristatelite/
├── README.md
├── pyproject.toml
├── Makefile
├── .gitignore
├── data/
│   ├── raw/.gitkeep
│   └── manifests/.gitkeep
├── scripts/
│   ├── download_nasa.py
│   └── inventory_nasa.py
├── src/tristatelite/
│   ├── __init__.py
│   ├── device.py
│   ├── provenance.py
│   └── data/
│       ├── __init__.py
│       ├── download.py
│       ├── archive_inventory.py
│       └── schema_report.py
└── tests/
    ├── test_device.py
    ├── test_download.py
    ├── test_archive_inventory.py
    └── test_schema_report.py
```

---

### Task 1: Initialize Repository and Reproducible Environment

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `Makefile`
- Create: `README.md`
- Create: `src/tristatelite/__init__.py`
- Create: `src/tristatelite/device.py`
- Create: `tests/test_device.py`

**Interfaces:**
- Produces: `select_device() -> torch.device`

- [ ] **Step 1: Initialize Git and directories**

Run:

```bash
git init
mkdir -p data/raw data/manifests scripts src/tristatelite/data tests
touch data/raw/.gitkeep data/manifests/.gitkeep
```

- [ ] **Step 2: Create `pyproject.toml`**

```toml
[project]
name = "tristatelite"
version = "0.1.0"
description = "Lightweight probabilistic joint SOC, SOH, and TTE prediction"
requires-python = ">=3.11,<3.13"
dependencies = [
  "torch>=2.2,<3.0",
  "numpy>=1.26,<3.0",
  "pandas>=2.2,<3.0",
  "scipy>=1.12,<2.0",
  "pyarrow>=15,<25",
  "pyyaml>=6,<7",
]

[project.optional-dependencies]
dev = [
  "pytest>=8,<10",
  "ruff>=0.5,<1.0",
]

[build-system]
requires = ["setuptools>=69"]
build-backend = "setuptools.build_meta"

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]

[tool.ruff]
line-length = 100
```

- [ ] **Step 3: Create `.gitignore`**

```gitignore
.venv/
__pycache__/
.pytest_cache/
.ruff_cache/
*.pyc
.DS_Store
data/raw/*
!data/raw/.gitkeep
data/manifests/*.json
data/manifests/*.md
data/manifests/*.csv
```

- [ ] **Step 4: Write the failing device test**

```python
# tests/test_device.py
from tristatelite.device import select_device


def test_select_device_returns_supported_device():
    assert select_device().type in {"mps", "cuda", "cpu"}
```

- [ ] **Step 5: Verify the test fails**

Run:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m pytest tests/test_device.py -v
```

Expected: FAIL because `tristatelite.device` does not exist.

- [ ] **Step 6: Implement device selection**

```python
# src/tristatelite/device.py
import torch


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
```

- [ ] **Step 7: Create minimal README and Makefile**

```makefile
install:
	python -m pip install -e '.[dev]'

test:
	python -m pytest -q

lint:
	ruff check src tests scripts

phase1: test lint
	python scripts/download_nasa.py
	python scripts/inventory_nasa.py
```

README must state that Phase 1 only downloads and inventories the archive and must stop before semantic parsing.

- [ ] **Step 8: Run verification**

```bash
python -m pytest tests/test_device.py -v
ruff check src tests
```

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml .gitignore Makefile README.md src tests data

git commit -m "chore: initialize TriStateLite data discovery repository"
```

---

### Task 2: Streaming Downloader with Provenance and Integrity Checks

**Files:**
- Create: `src/tristatelite/provenance.py`
- Create: `src/tristatelite/data/download.py`
- Create: `scripts/download_nasa.py`
- Create: `tests/test_download.py`

**Interfaces:**
- Produces: `sha256_file(path: Path) -> str`
- Produces: `download_file(url: str, destination: Path, force: bool = False) -> dict[str, object]`
- Produces file: `data/manifests/nasa_download.json`

- [ ] **Step 1: Write SHA256 test**

```python
# tests/test_download.py
import hashlib
from pathlib import Path
from tristatelite.provenance import sha256_file


def test_sha256_file_matches_hashlib(tmp_path: Path):
    path = tmp_path / "x.bin"
    path.write_bytes(b"battery-data")
    expected = hashlib.sha256(b"battery-data").hexdigest()
    assert sha256_file(path) == expected
```

- [ ] **Step 2: Write downloader tests using a local HTTP server**

Use Python’s `http.server.ThreadingHTTPServer` in a pytest fixture. Tests must assert:

1. downloaded bytes equal served bytes;
2. destination is absent while only `.part` exists during an interrupted transfer;
3. an existing non-empty destination raises `FileExistsError` unless `force=True`;
4. returned provenance contains source URL, byte count, SHA256, and an ISO-8601 UTC timestamp;
5. `.part` is removed after a failed request.

- [ ] **Step 3: Run tests and confirm failure**

```bash
python -m pytest tests/test_download.py -v
```

Expected: missing modules/functions.

- [ ] **Step 4: Implement `sha256_file`**

Read in 8 MiB blocks and return lowercase hexadecimal SHA256.

- [ ] **Step 5: Implement `download_file`**

Required behavior:

```python
from pathlib import Path

provenance = download_file(
    url="https://data.nasa.gov/docs/legacy/battery_alt_dataset.zip",
    destination=Path("data/raw/battery_alt_dataset.zip"),
    force=False,
)
```

Implementation requirements:

- use `urllib.request.urlopen` so no extra HTTP dependency is needed;
- set a descriptive User-Agent;
- use timeout 60 seconds;
- stream 8 MiB blocks;
- print cumulative MiB after each block;
- write to `battery_alt_dataset.zip.part`;
- flush and `os.fsync()` before rename;
- atomically replace destination only after complete download;
- clean partial file on any exception;
- return JSON-serializable provenance.

- [ ] **Step 6: Implement CLI**

`scripts/download_nasa.py` must support:

```bash
python scripts/download_nasa.py \
  --url https://data.nasa.gov/docs/legacy/battery_alt_dataset.zip \
  --output data/raw/battery_alt_dataset.zip \
  --manifest data/manifests/nasa_download.json
```

Defaults must be the three paths above. Add `--force` as a boolean flag.

- [ ] **Step 7: Run unit tests**

```bash
python -m pytest tests/test_download.py -v
ruff check src tests scripts
```

Expected: PASS.

- [ ] **Step 8: Download the real NASA archive**

```bash
python scripts/download_nasa.py
```

Expected outputs:

```text
data/raw/battery_alt_dataset.zip
data/manifests/nasa_download.json
```

- [ ] **Step 9: Verify the real archive**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path
from tristatelite.provenance import sha256_file

archive = Path('data/raw/battery_alt_dataset.zip')
manifest = json.loads(Path('data/manifests/nasa_download.json').read_text())
assert archive.is_file() and archive.stat().st_size > 0
assert manifest['sha256'] == sha256_file(archive)
assert manifest['bytes'] == archive.stat().st_size
print(manifest)
PY
```

Expected: assertions pass and manifest prints.

- [ ] **Step 10: Commit code and manifest metadata**

Do not commit the ZIP. Commit the manifest because it records provenance.

```bash
git add src/tristatelite/provenance.py src/tristatelite/data/download.py \
  scripts/download_nasa.py tests/test_download.py data/manifests/nasa_download.json

git commit -m "feat: download NASA battery archive with provenance"
```

---

### Task 3: Safe ZIP Inventory and File-Type Probing

**Files:**
- Create: `src/tristatelite/data/archive_inventory.py`
- Create: `tests/test_archive_inventory.py`

**Interfaces:**
- Produces: `inventory_zip(path: Path) -> dict[str, object]`
- Produces each member record with `path`, `suffix`, `compressed_bytes`, `uncompressed_bytes`, `crc`, `is_encrypted`, `unsafe_path`

- [ ] **Step 1: Write a synthetic ZIP test**

```python
# tests/test_archive_inventory.py
import zipfile
from pathlib import Path
from tristatelite.data.archive_inventory import inventory_zip


def test_inventory_zip_counts_members_and_extensions(tmp_path: Path):
    path = tmp_path / "sample.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("pack01/cycle01.csv", "time,voltage,current\n0,4.2,-1\n")
        zf.writestr("README.txt", "notes")
    report = inventory_zip(path)
    assert report["member_count"] == 2
    assert report["extensions"] == {".csv": 1, ".txt": 1}
    assert report["total_uncompressed_bytes"] > 0
    assert all(not item["unsafe_path"] for item in report["members"])
```

- [ ] **Step 2: Add path traversal and encrypted-member tests**

Create a member named `../escape.csv` and assert `unsafe_path=True`. If the library marks encryption, record it and refuse later probing.

- [ ] **Step 3: Run tests and confirm failure**

```bash
python -m pytest tests/test_archive_inventory.py -v
```

- [ ] **Step 4: Implement safe ZIP inventory**

Rules:

- validate with `zipfile.is_zipfile`;
- normalize suffix to lowercase;
- detect absolute paths and `..` traversal;
- never extract during inventory;
- record archive SHA256 and size;
- sort members lexicographically;
- count extensions;
- record top-level directory token counts;
- record member filename token frequencies after splitting on `/`, `_`, `-`, and `.`;
- return JSON-serializable values only.

- [ ] **Step 5: Run tests**

```bash
python -m pytest tests/test_archive_inventory.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/tristatelite/data/archive_inventory.py tests/test_archive_inventory.py
git commit -m "feat: inventory NASA ZIP safely"
```

---

### Task 4: Schema Probes for Text, CSV, and MATLAB Files

**Files:**
- Create: `src/tristatelite/data/schema_report.py`
- Create: `tests/test_schema_report.py`

**Interfaces:**
- Produces: `probe_member(archive: Path, member_path: str) -> dict[str, object]`
- Produces: `build_schema_report(archive: Path) -> dict[str, object]`

- [ ] **Step 1: Write CSV probe test**

Synthetic CSV report must include:

- encoding candidate;
- delimiter;
- header names;
- first five rows as strings;
- row-width consistency;
- candidate semantic scores for time, voltage, current, temperature, cycle, battery, phase, and capacity.

- [ ] **Step 2: Write MATLAB probe test**

Create a classic MAT file with `scipy.io.savemat` containing arrays named `time`, `voltage`, and `current`; place it into a ZIP. Assert the probe reports variable names, shapes and dtypes without loading full numeric arrays into JSON.

- [ ] **Step 3: Run tests and confirm failure**

```bash
python -m pytest tests/test_schema_report.py -v
```

- [ ] **Step 4: Implement text/CSV probing**

For `.csv`, `.tsv`, `.txt`, `.dat`:

- read at most 64 KiB;
- try UTF-8, UTF-8-SIG, then Latin-1;
- infer delimiter among comma, tab, semicolon and whitespace;
- inspect at most 20 rows;
- do not treat any inferred semantic role as confirmed.

Candidate-role scoring is keyword-only and must emit both score and evidence. Use these normalized keyword sets:

```python
ROLE_KEYWORDS = {
    "time": {"time", "timestamp", "seconds", "sec", "t"},
    "voltage": {"voltage", "volt", "v"},
    "current": {"current", "amp", "amps", "a", "i"},
    "temperature": {"temperature", "temp", "celsius", "degc"},
    "cycle": {"cycle", "cycle_id", "cycleindex"},
    "battery": {"battery", "pack", "cell", "battery_id", "cell_id"},
    "phase": {"phase", "mode", "type", "state", "operation"},
    "capacity": {"capacity", "cap", "ah", "amp_hour"},
}
```

A one-character keyword matches only an exact normalized header, never a substring.

- [ ] **Step 5: Implement classic MATLAB probing**

For `.mat`:

- extract only that member to a `TemporaryDirectory`;
- call `scipy.io.whosmat`;
- return variable names, shapes, and MATLAB class;
- if classic MAT probing fails because the file is HDF5-based, record `probe_status="hdf5_mat_requires_phase2_support"` rather than crashing or installing new dependencies.

- [ ] **Step 6: Implement report aggregation**

`build_schema_report()` must:

- call `inventory_zip()`;
- probe all supported members when there are ≤200;
- otherwise probe up to 20 members per extension and every README-like file;
- group files by common path and filename patterns;
- rank candidate files for time-series data by presence of at least time + voltage + current candidates;
- mark all candidate roles `status="unconfirmed"`;
- include unsupported extension counts.

- [ ] **Step 7: Run tests and commit**

```bash
python -m pytest tests/test_schema_report.py -v
ruff check src tests

git add src/tristatelite/data/schema_report.py tests/test_schema_report.py
git commit -m "feat: probe NASA archive schemas without semantic assumptions"
```

---

### Task 5: Generate Machine-Readable and Human-Readable NASA Reports

**Files:**
- Create: `scripts/inventory_nasa.py`
- Modify: `README.md`

**Interfaces:**
- Produces: `data/manifests/nasa_archive_inventory.json`
- Produces: `data/manifests/nasa_schema_report.json`
- Produces: `data/manifests/nasa_schema_report.md`

- [ ] **Step 1: Implement CLI arguments**

```bash
python scripts/inventory_nasa.py \
  --archive data/raw/battery_alt_dataset.zip \
  --inventory-json data/manifests/nasa_archive_inventory.json \
  --schema-json data/manifests/nasa_schema_report.json \
  --schema-markdown data/manifests/nasa_schema_report.md
```

- [ ] **Step 2: Implement Markdown rendering**

The report must contain these sections in order:

1. Archive provenance: path, bytes, SHA256;
2. Extension counts;
3. Top-level directory counts;
4. Candidate README/documentation files;
5. Candidate time-series files;
6. Candidate semantic fields with evidence and unconfirmed status;
7. MATLAB variable inventories;
8. Unsafe/encrypted/unsupported members;
9. Explicit human decisions required;
10. Phase 1 stop notice.

The “human decisions required” table must ask for exactly:

```text
battery_id source
cycle_id source or cycle-boundary rule
time field and unit
voltage field and unit
current field, unit, and sign convention
temperature field and unit, or confirmation that it is absent
phase field or phase-inference rule
capacity field or confirmation that it must be integrated
complete-discharge termination rule
```

- [ ] **Step 3: Add report consistency assertions**

Before writing:

- inventory SHA256 must equal download-manifest SHA256 when the manifest exists;
- every probed member must exist in inventory;
- every candidate role must remain `unconfirmed`;
- unsafe/encrypted members must not be probed.

- [ ] **Step 4: Run the real report generation**

```bash
python scripts/inventory_nasa.py
```

- [ ] **Step 5: Run complete verification**

```bash
python -m pytest -q
ruff check src tests scripts
python - <<'PY'
import json
from pathlib import Path
for name in [
    'nasa_download.json',
    'nasa_archive_inventory.json',
    'nasa_schema_report.json',
]:
    path = Path('data/manifests') / name
    assert path.is_file() and path.stat().st_size > 0
    json.loads(path.read_text())
md = Path('data/manifests/nasa_schema_report.md')
assert md.is_file() and 'Phase 1 stop' in md.read_text()
print('Phase 1 artifacts verified')
PY
```

Expected: all tests pass, lint passes, four manifest/report files exist.

- [ ] **Step 6: Update README with exact commands**

README quick start:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
python scripts/download_nasa.py
python scripts/inventory_nasa.py
```

State clearly: “Stop here and review `data/manifests/nasa_schema_report.md`; do not build labels or train a model until the field mapping and cycle semantics are confirmed.”

- [ ] **Step 7: Commit**

```bash
git add scripts/inventory_nasa.py README.md data/manifests/*.json data/manifests/*.md
git commit -m "docs: publish NASA schema discovery report"
```

---

## Phase 1 Completion Gate

Phase 1 is complete only when all of the following are true:

```bash
python -m pytest -q
ruff check src tests scripts
git status --short
```

Required evidence:

- tests and lint pass;
- archive exists locally but is not tracked by Git;
- `nasa_download.json` records real bytes and SHA256;
- `nasa_archive_inventory.json` lists every member;
- `nasa_schema_report.json` is valid JSON;
- `nasa_schema_report.md` identifies candidate schemas without confirming them;
- Git working tree is clean;
- no files for labels, splits, windows, models, losses, training, or evaluation have been created.

## Handoff for Phase 2

Return these report excerpts to the project owner:

1. archive extension counts;
2. top-level directories;
3. five highest-ranked time-series candidates;
4. their headers or MAT variables and shapes;
5. likely battery and cycle identifiers;
6. evidence for time, voltage, current, temperature, phase, and capacity candidates;
7. unresolved ambiguities.

A new Phase 2 implementation plan will then define the exact NASA adapter, cycle reconstruction, labels, leakage-safe split, window dataset, baselines, TriStateLite model, training, and evaluation using the observed schema rather than assumptions.
