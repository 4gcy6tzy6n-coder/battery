# TriStateLite Master Implementation Roadmap

> This roadmap defines the complete project sequence. Only the separate Phase 1 implementation plan is executable before the NASA archive has been inspected; Phase 2 must be regenerated from the real schema report.

**Goal:** Build a reproducible, leakage-safe, Mac-trainable pipeline for joint probabilistic SOC, SOH, and TTE prediction from public battery time-series data.

**Architecture:** The repository separates source-specific ingestion from a canonical battery-cycle schema. A causal window dataset feeds a fast TCN encoder and a slow historical-cycle GRU encoder; gated fusion drives ordered quantile heads for SOC, SOH, and log-TTE, with an optional load-conditioned physics-consistency loss.

**Tech Stack:** Python 3.11, PyTorch, NumPy, pandas, SciPy, scikit-learn, PyArrow, PyYAML, pytest, matplotlib.

## Global Constraints

- Train on Apple Silicon Mac using float32; select MPS, then CUDA, then CPU.
- Model parameter count must be below 500,000.
- Use battery-level splits only; no battery ID may appear in more than one split.
- Fit preprocessing statistics on the training split only.
- Use only causal inputs; no bidirectional models or future interpolation.
- Do not assume NASA archive field names before generating and reviewing the archive inventory.
- Default sequence rate is 1 Hz, fast window length is 128 samples, slow history length is 8 completed cycles.
- First executable milestone is NASA data audit → canonicalization → labels → split → baseline → TriStateLite one-seed run.
- CALCE support is a separate final milestone and must not delay the NASA MVP.

---

## Repository Map

```text
tristatelite/
├── README.md
├── pyproject.toml
├── Makefile
├── configs/
│   ├── data/nasa_randomized.yaml
│   ├── data/calce.yaml
│   ├── model/gru_multitask.yaml
│   ├── model/tristatelite.yaml
│   └── train/default.yaml
├── data/
│   ├── raw/.gitkeep
│   ├── interim/.gitkeep
│   ├── processed/.gitkeep
│   └── manifests/.gitkeep
├── docs/
│   ├── data_dictionary.md
│   └── leakage_contract.md
├── scripts/
│   ├── download_nasa.py
│   ├── inventory_archive.py
│   ├── build_dataset.py
│   ├── train.py
│   ├── evaluate.py
│   └── run_smoke.sh
├── src/tristatelite/
│   ├── __init__.py
│   ├── config.py
│   ├── device.py
│   ├── schemas.py
│   ├── data/
│   │   ├── nasa_inventory.py
│   │   ├── nasa_adapter.py
│   │   ├── calce_adapter.py
│   │   ├── canonicalize.py
│   │   ├── labels.py
│   │   ├── split.py
│   │   ├── scaling.py
│   │   ├── windows.py
│   │   └── audit.py
│   ├── models/
│   │   ├── causal_tcn.py
│   │   ├── history_gru.py
│   │   ├── fusion.py
│   │   ├── quantile_heads.py
│   │   ├── gru_baseline.py
│   │   └── tristatelite.py
│   ├── losses/
│   │   ├── pinball.py
│   │   └── consistency.py
│   ├── training/
│   │   ├── engine.py
│   │   ├── checkpoint.py
│   │   └── seed.py
│   └── evaluation/
│       ├── metrics.py
│       ├── calibration.py
│       ├── latency.py
│       └── reports.py
├── tests/
│   ├── fixtures/synthetic_cycle.py
│   ├── test_inventory.py
│   ├── test_canonicalize.py
│   ├── test_labels.py
│   ├── test_split.py
│   ├── test_scaling.py
│   ├── test_windows.py
│   ├── test_quantile_heads.py
│   ├── test_losses.py
│   ├── test_model.py
│   ├── test_metrics.py
│   └── test_smoke_training.py
└── outputs/.gitkeep
```

## Canonical Interfaces

```python
# src/tristatelite/schemas.py
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

@dataclass(frozen=True)
class CanonicalSample:
    battery_id: str
    cycle_id: str
    timestamp_s: float
    voltage_v: float
    current_a: float          # discharge is positive
    temperature_c: float
    temperature_available: bool
    phase: Literal["charge", "discharge", "rest", "unknown"]

@dataclass(frozen=True)
class CycleSummary:
    battery_id: str
    cycle_id: str
    cycle_index: int
    start_s: float
    end_s: float
    delivered_ah: float
    discharge_duration_s: float
    mean_voltage_v: float
    voltage_slope_mean: float
    mean_current_a: float
    current_std_a: float
    mean_temperature_c: float
    temperature_rise_c: float

@dataclass(frozen=True)
class SplitManifest:
    split_id: str
    train_batteries: tuple[str, ...]
    val_batteries: tuple[str, ...]
    test_batteries: tuple[str, ...]
```

The canonical processed sample table must contain:

```text
battery_id, cycle_id, cycle_index, timestamp_s,
voltage_v, current_a, temperature_c, temperature_available, phase,
soc_target, soh_target, tte_seconds, log_tte_target,
q_ref_ah, i_eff_60s, current_cv_60s
```

---

### Task 1: Repository, Environment, and Deterministic Device Selection

**Files:**
- Create: `pyproject.toml`
- Create: `Makefile`
- Create: `src/tristatelite/__init__.py`
- Create: `src/tristatelite/device.py`
- Create: `src/tristatelite/training/seed.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Produces: `select_device() -> torch.device`
- Produces: `seed_everything(seed: int) -> None`

- [ ] **Step 1: Create the package metadata**

Use this dependency set in `pyproject.toml`:

```toml
[project]
name = "tristatelite"
version = "0.1.0"
requires-python = ">=3.11,<3.13"
dependencies = [
  "torch>=2.2,<3.0",
  "numpy>=1.26,<3.0",
  "pandas>=2.2,<3.0",
  "scipy>=1.12,<2.0",
  "scikit-learn>=1.4,<2.0",
  "pyarrow>=15,<25",
  "pyyaml>=6,<7",
  "matplotlib>=3.8,<4.0",
]

[project.optional-dependencies]
dev = ["pytest>=8,<10", "ruff>=0.5,<1.0"]

[build-system]
requires = ["setuptools>=69"]
build-backend = "setuptools.build_meta"

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]

[tool.ruff]
line-length = 100
```

- [ ] **Step 2: Write failing tests for device and seeding**

```python
# tests/test_model.py
import numpy as np
import torch
from tristatelite.device import select_device
from tristatelite.training.seed import seed_everything

def test_select_device_returns_supported_type():
    assert select_device().type in {"mps", "cuda", "cpu"}

def test_seed_everything_is_reproducible():
    seed_everything(7)
    a = np.random.rand(3)
    b = torch.rand(3)
    seed_everything(7)
    assert np.allclose(a, np.random.rand(3))
    assert torch.equal(b, torch.rand(3))
```

- [ ] **Step 3: Verify tests fail**

Run: `python -m pytest tests/test_model.py -v`

Expected: import failures for `tristatelite.device` and `tristatelite.training.seed`.

- [ ] **Step 4: Implement device and seed helpers**

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

```python
# src/tristatelite/training/seed.py
import os
import random
import numpy as np
import torch

def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

- [ ] **Step 5: Run tests and lint**

Run: `python -m pytest tests/test_model.py -v && ruff check src tests`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml Makefile src tests/test_model.py
git commit -m "chore: initialize TriStateLite package"
```

---

### Task 2: NASA Download and Archive Inventory Gate

**Files:**
- Create: `scripts/download_nasa.py`
- Create: `scripts/inventory_archive.py`
- Create: `src/tristatelite/data/nasa_inventory.py`
- Create: `tests/test_inventory.py`
- Create: `data/raw/.gitkeep`
- Create: `data/manifests/.gitkeep`

**Interfaces:**
- Produces: `download_file(url: str, destination: Path, sha256: str | None) -> Path`
- Produces: `inventory_archive(path: Path) -> dict[str, object]`
- Produces file: `data/manifests/nasa_archive_inventory.json`

- [ ] **Step 1: Write an inventory test using a synthetic ZIP**

```python
# tests/test_inventory.py
import json
import zipfile
from pathlib import Path
from tristatelite.data.nasa_inventory import inventory_archive

def test_inventory_archive_lists_extensions_and_members(tmp_path: Path):
    archive = tmp_path / "sample.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("pack01/cycle01.csv", "time,voltage,current\n0,4.2,0.5\n")
        zf.writestr("README.txt", "schema notes")
    report = inventory_archive(archive)
    assert report["member_count"] == 2
    assert report["extensions"][".csv"] == 1
    assert report["extensions"][".txt"] == 1
    assert report["members"][0]["path"]
    json.dumps(report)
```

- [ ] **Step 2: Run the test and confirm failure**

Run: `python -m pytest tests/test_inventory.py -v`

Expected: missing module/function.

- [ ] **Step 3: Implement archive inventory without parsing semantics**

`inventory_archive()` must:

1. calculate archive SHA256;
2. list every non-directory member;
3. record path, suffix, compressed size, uncompressed size;
4. count suffixes case-insensitively;
5. inspect only the first 8 KB of text-like files for delimiter and header candidates;
6. never extract executable files;
7. return JSON-serializable values.

- [ ] **Step 4: Implement streaming download with temporary-file rename**

The script must default to the official resource URL:

```text
https://data.nasa.gov/docs/legacy/battery_alt_dataset.zip
```

Required behavior:

```bash
python scripts/download_nasa.py \
  --url https://data.nasa.gov/docs/legacy/battery_alt_dataset.zip \
  --output data/raw/battery_alt_dataset.zip
```

- write to `*.part`;
- stream in 8 MiB chunks;
- print downloaded MiB;
- remove partial file on HTTP error;
- optionally verify `--sha256`;
- never overwrite an existing non-empty archive unless `--force` is set.

- [ ] **Step 5: Generate the real inventory**

Run:

```bash
python scripts/download_nasa.py --output data/raw/battery_alt_dataset.zip
python scripts/inventory_archive.py \
  --archive data/raw/battery_alt_dataset.zip \
  --output data/manifests/nasa_archive_inventory.json
```

Expected: JSON contains archive hash, member count, extension counts and candidate table headers.

- [ ] **Step 6: Stop for human review of the inventory**

Do not implement the NASA semantic adapter until the report answers:

- Which file formats contain time series?
- Which directory/file token identifies battery or pack?
- Which fields correspond to time, voltage, current and temperature?
- Are cycle boundaries explicit, or must they be derived?
- Is capacity supplied, or must it be integrated?

Record the confirmed mapping in `configs/data/nasa_randomized.yaml`; do not infer silently.

- [ ] **Step 7: Commit**

```bash
git add scripts/download_nasa.py scripts/inventory_archive.py \
  src/tristatelite/data/nasa_inventory.py tests/test_inventory.py data/manifests

git commit -m "feat: add NASA archive acquisition and inventory gate"
```

---

### Task 3: Canonical Schema and Source-Specific NASA Adapter

**Files:**
- Create: `src/tristatelite/schemas.py`
- Create: `src/tristatelite/data/nasa_adapter.py`
- Create: `src/tristatelite/data/canonicalize.py`
- Create: `configs/data/nasa_randomized.yaml`
- Create: `tests/fixtures/synthetic_cycle.py`
- Create: `tests/test_canonicalize.py`

**Interfaces:**
- Consumes: reviewed field mapping from Task 2
- Produces: `load_nasa_records(root: Path, config: dict) -> pd.DataFrame`
- Produces: `canonicalize_samples(raw: pd.DataFrame, config: dict) -> pd.DataFrame`
- Produces canonical columns defined in the repository interface section

- [ ] **Step 1: Create a synthetic raw NASA-like fixture**

```python
# tests/fixtures/synthetic_cycle.py
import pandas as pd

def synthetic_raw_cycle() -> pd.DataFrame:
    return pd.DataFrame({
        "pack": ["P01"] * 5,
        "cycle": ["C001"] * 5,
        "time_raw": [0.0, 0.7, 1.8, 3.1, 4.0],
        "voltage_raw": [4.2, 4.18, 4.15, 4.1, 4.05],
        "current_raw": [-1.0, -1.0, -1.2, -1.2, -1.1],
        "temp_raw": [25.0, 25.0, 25.1, 25.1, 25.2],
        "mode": ["D"] * 5,
    })
```

- [ ] **Step 2: Write failing canonicalization tests**

The test must assert:

- output timestamps are integer seconds at 1 Hz;
- discharge current is positive;
- missing seconds are forward-filled only;
- no output timestamp is created before the first observation;
- required columns are present;
- timestamps are strictly increasing within battery/cycle.

- [ ] **Step 3: Implement `CanonicalSample` and `CycleSummary` dataclasses**

Use the exact signatures in the interface section.

- [ ] **Step 4: Implement YAML-driven field mapping**

The production adapter configuration must be generated only after the Phase 1 archive report has been reviewed. Phase 2 will define the exact file glob, format, path rules, field names, current-sign convention, and phase mapping from the observed archive. This roadmap deliberately does not invent those values.

- [ ] **Step 4: Generate the production mapping from the reviewed Phase 1 report**

Create `configs/data/nasa_randomized.yaml` from the actual schema report and add a fixture-specific `configs/data/nasa_synthetic_test.yaml` for unit tests. The production build command must validate that every source field resolves to a real archive member or real column before processing any sample.

- [ ] **Step 5: Implement causal 1 Hz canonicalization**

Algorithm per battery/cycle:

1. sort by raw timestamp;
2. reject duplicate timestamps with conflicting values;
3. floor timestamp to integer second;
4. keep the last observation in each second;
5. reindex from first to last observed second;
6. forward-fill at most 5 seconds;
7. remove rows remaining missing in voltage/current;
8. normalize current sign so discharge is positive;
9. infer `temperature_available` before filling temperature;
10. fill unavailable temperature later using training statistics, not here.

- [ ] **Step 6: Run tests**

Run: `python -m pytest tests/test_canonicalize.py -v`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/tristatelite/schemas.py src/tristatelite/data \
  configs/data/nasa_randomized.yaml tests

git commit -m "feat: canonicalize NASA battery time series"
```

---

### Task 4: Cycle Detection, Capacity Integration, and Label Generation

**Files:**
- Create: `src/tristatelite/data/labels.py`
- Create: `tests/test_labels.py`
- Create: `docs/data_dictionary.md`

**Interfaces:**
- Produces: `integrate_discharge_ah(time_s: np.ndarray, current_a: np.ndarray) -> float`
- Produces: `add_cycle_targets(samples: pd.DataFrame, protocol: str) -> tuple[pd.DataFrame, pd.DataFrame]`
- Returns: labeled sample table and cycle-summary table

- [ ] **Step 1: Write exact numeric tests for charge integration**

```python
import numpy as np
from tristatelite.data.labels import integrate_discharge_ah

def test_one_amp_for_one_hour_is_one_ah():
    time_s = np.arange(0, 3601, dtype=float)
    current_a = np.ones_like(time_s)
    assert abs(integrate_discharge_ah(time_s, current_a) - 1.0) < 1e-4
```

- [ ] **Step 2: Write target tests on a ten-second synthetic discharge**

Assert:

- first SOC is 1.0 within tolerance;
- final SOC is 0.0 within tolerance;
- TTE decreases monotonically to 0;
- `log_tte_target == log1p(tte_seconds)`;
- SOH is constant within one cycle;
- current cycle final capacity is absent from input feature columns;
- slow history for cycle k contains only cycles with index < k.

- [ ] **Step 3: Implement trapezoidal Ah integration**

Only integrate rows whose phase is `discharge` and current is positive. Reject time reversals and cycles shorter than 60 seconds.

- [ ] **Step 4: Implement complete-cycle validity rules**

A discharge cycle is valid only when:

- duration ≥ 60 s;
- at least 50 canonical samples;
- delivered capacity > 0;
- voltage has a net downward trend or source metadata explicitly marks a complete discharge;
- no gap larger than configured limit remains.

Write rejected cycles with reason codes to `data/manifests/rejected_cycles.csv`.

- [ ] **Step 5: Implement calibrated and zero-shot Q_ref protocols**

`calibrated`:

- sort complete cycles by cycle index;
- use median capacity of first three valid cycles as battery-specific Q_ref;
- mark those three cycles `is_calibration_cycle=True`;
- exclude calibration cycles from test metrics.

`zero_shot`:

- compute global Q_ref only from training batteries’ first three valid cycles;
- serialize it in the fitted preprocessing artifact;
- apply the same scalar to validation/test batteries.

- [ ] **Step 6: Implement SOC, SOH and TTE targets**

Use cumulative trapezoidal discharge Ah. Clip SOC and SOH only after logging unclipped values for audit. If more than 1% of labels require clipping beyond `[-0.02, 1.05]`, fail the build.

- [ ] **Step 7: Generate cycle summaries**

Summaries may include final delivered capacity only for cycles completed before the current sample’s cycle. Save to `data/processed/cycle_summaries.parquet`.

- [ ] **Step 8: Run tests and commit**

```bash
python -m pytest tests/test_labels.py -v
git add src/tristatelite/data/labels.py tests/test_labels.py docs/data_dictionary.md
git commit -m "feat: generate leakage-safe SOC SOH and TTE targets"
```

---

### Task 5: Battery-Level Split Manifest and Leakage Audit

**Files:**
- Create: `src/tristatelite/data/split.py`
- Create: `src/tristatelite/data/audit.py`
- Create: `tests/test_split.py`
- Create: `docs/leakage_contract.md`

**Interfaces:**
- Produces: `make_group_split(battery_ids: list[str], seed: int, fractions: tuple[float, float, float]) -> SplitManifest`
- Produces: `audit_split(manifest: SplitManifest, samples: pd.DataFrame) -> dict[str, object]`
- Produces file: `data/manifests/split_seed_2026.json`

- [ ] **Step 1: Write split isolation tests**

Use 26 synthetic battery IDs. Assert:

- train/val/test sets are pairwise disjoint;
- union equals all IDs;
- repeated calls with seed 2026 are identical;
- each split is non-empty;
- approximate sizes are 18/4/4;
- every sample inherits split from battery ID only.

- [ ] **Step 2: Implement deterministic grouped split**

Use a local NumPy generator. Sort IDs before permutation. If reviewed metadata provides load-condition groups, stratify approximately by group; otherwise record `stratification_used=false` in the manifest.

- [ ] **Step 3: Implement leakage audit**

Fail on:

- battery overlap;
- missing battery assignment;
- scaler fit rows outside train;
- slow-history cycle index not lower than current cycle index;
- any feature column containing `target`, `future`, `final_capacity`, or current-cycle delivered capacity;
- duplicate `(battery_id, cycle_id, timestamp_s)` across splits.

- [ ] **Step 4: Run and commit**

```bash
python -m pytest tests/test_split.py -v
git add src/tristatelite/data/split.py src/tristatelite/data/audit.py \
  tests/test_split.py docs/leakage_contract.md
git commit -m "feat: enforce battery-level split and leakage audit"
```

---

### Task 6: Train-Only Scaling, Causal Feature Engineering, and Window Cache

**Files:**
- Create: `src/tristatelite/data/scaling.py`
- Create: `src/tristatelite/data/windows.py`
- Create: `tests/test_scaling.py`
- Create: `tests/test_windows.py`

**Interfaces:**
- Produces: `fit_scaler(train_df: pd.DataFrame, feature_names: list[str]) -> dict[str, np.ndarray]`
- Produces: `transform_features(df: pd.DataFrame, artifact: dict) -> pd.DataFrame`
- Produces: `BatteryWindowDataset`
- Dataset item keys: `fast_x`, `slow_x`, `slow_mask`, `targets`, `physics`, `metadata`

- [ ] **Step 1: Write a train-only scaling test**

Create train values centered at 0 and test values centered at 100. Assert scaler mean equals train mean and transformed test mean remains far from zero.

- [ ] **Step 2: Implement robust standardization**

For each feature, store train median and IQR. Transform as `(x - median) / max(IQR, 1e-6)`, clip to `[-10, 10]`. For unavailable temperature, fill with training median and preserve the availability mask.

- [ ] **Step 3: Write causal feature tests**

For a sequence where future current jumps at t=100, assert features at t<100 are identical whether future rows are present or removed.

- [ ] **Step 4: Implement causal rolling features**

Use only backward-looking windows for:

- deltas;
- current mean/std at 10 s;
- current mean and coefficient of variation at 60 s;
- voltage slope at 30 s;
- temperature slope at 60 s.

- [ ] **Step 5: Implement `BatteryWindowDataset`**

Constructor:

```python
class BatteryWindowDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        samples: pd.DataFrame,
        cycle_summaries: pd.DataFrame,
        battery_ids: set[str],
        feature_names: list[str],
        fast_length: int = 128,
        history_length: int = 8,
        stride: int = 10,
    ) -> None: ...
```

Behavior:

- left-pad early fast windows with zeros and return `fast_mask`;
- select only previous completed cycle summaries;
- left-pad slow history and return `slow_mask`;
- use stride 10 for training and stride 1 for evaluation;
- exclude charge/rest samples from prediction dataset;
- include metadata battery/cycle/time for grouped reporting.

- [ ] **Step 6: Save cache with provenance**

Cache filename must include source archive SHA256 prefix, split ID, window length, stride and config hash. Reject cache when any value differs.

- [ ] **Step 7: Run and commit**

```bash
python -m pytest tests/test_scaling.py tests/test_windows.py -v
git add src/tristatelite/data/scaling.py src/tristatelite/data/windows.py tests
git commit -m "feat: build causal scaled battery windows"
```

---

### Task 7: Ordered Quantile Heads and Losses

**Files:**
- Create: `src/tristatelite/models/quantile_heads.py`
- Create: `src/tristatelite/losses/pinball.py`
- Create: `src/tristatelite/losses/consistency.py`
- Create: `tests/test_quantile_heads.py`
- Create: `tests/test_losses.py`

**Interfaces:**
- Produces: `BoundedQuantileHead(in_dim: int)`
- Produces: `PositiveQuantileHead(in_dim: int)`
- Produces: `pinball_loss(pred: Tensor, target: Tensor, quantiles: Tensor) -> Tensor`
- Produces: `physics_consistency_loss(...) -> Tensor`

- [ ] **Step 1: Write ordering tests**

For 1,000 random hidden vectors, assert every head output has shape `[1000, 3]` and satisfies `q05 <= q50 <= q95`. For bounded heads assert values lie in `[0,1]`.

- [ ] **Step 2: Implement bounded and positive ordered heads**

Use the exact formulas from the design spec. Do not sort outputs after prediction because sorting breaks head semantics.

- [ ] **Step 3: Write pinball loss numeric test**

For exact prediction, loss must be zero. For a known under-prediction, verify the manual quantile-weighted value.

- [ ] **Step 4: Implement pinball loss**

Input prediction shape `[batch, 3]`, target shape `[batch]`, quantiles `[0.05, 0.50, 0.95]`. Return batch and quantile mean.

- [ ] **Step 5: Write physics consistency tests**

Assert:

- loss is near zero when model TTE equals physical TTE;
- loss is masked when mean current < 0.05 A;
- loss weight is lower for high current CV than stable current;
- gradients reach SOC, SOH and TTE medians.

- [ ] **Step 6: Implement consistency loss in log seconds**

Clamp only the physical TTE used in loss to `[0, 7 * 24 * 3600]` seconds. Return both scalar loss and diagnostics: active fraction, mean weight, median physical TTE.

- [ ] **Step 7: Run and commit**

```bash
python -m pytest tests/test_quantile_heads.py tests/test_losses.py -v
git add src/tristatelite/models/quantile_heads.py src/tristatelite/losses tests
git commit -m "feat: add ordered probabilistic heads and consistency loss"
```

---

### Task 8: Baseline Models

**Files:**
- Create: `src/tristatelite/models/gru_baseline.py`
- Create: `configs/model/gru_multitask.yaml`
- Extend: `tests/test_model.py`

**Interfaces:**
- Produces: `SingleTaskGRU(input_dim: int, hidden_dim: int, task: str)`
- Produces: `SharedMultiTaskGRU(input_dim: int, hidden_dim: int)`
- Both return dict keys `soc_q`, `soh_q`, `log_tte_q` as applicable

- [ ] **Step 1: Write forward-shape tests**

Use batch 4, length 128, input dim 12. Assert output shapes and quantile ordering.

- [ ] **Step 2: Implement single-task GRU**

One unidirectional GRU layer, hidden size 48, dropout 0, final valid hidden state selected using fast mask, one task-specific quantile head.

- [ ] **Step 3: Implement shared multi-task GRU**

One unidirectional GRU layer, hidden size 64, shared 64→64 MLP, three ordered quantile heads.

- [ ] **Step 4: Add parameter-count assertion**

Both baselines must be below 250,000 parameters.

- [ ] **Step 5: Run and commit**

```bash
python -m pytest tests/test_model.py -v
git add src/tristatelite/models/gru_baseline.py configs/model/gru_multitask.yaml tests/test_model.py
git commit -m "feat: add single-task and shared GRU baselines"
```

---

### Task 9: TriStateLite Fast Encoder, Slow Encoder, and Gated Fusion

**Files:**
- Create: `src/tristatelite/models/causal_tcn.py`
- Create: `src/tristatelite/models/history_gru.py`
- Create: `src/tristatelite/models/fusion.py`
- Create: `src/tristatelite/models/tristatelite.py`
- Create: `configs/model/tristatelite.yaml`
- Extend: `tests/test_model.py`

**Interfaces:**
- Produces: `CausalTCN(input_dim: int, channels: tuple[int, ...], kernel_size: int)`
- Produces: `HistoryGRU(input_dim: int, hidden_dim: int)`
- Produces: `GatedFusion(fast_dim: int, slow_dim: int, out_dim: int)`
- Produces: `TriStateLite(...) -> dict[str, Tensor]`

- [ ] **Step 1: Write a causal invariance test**

Create two sequences identical through t=63 and different afterward. Run the TCN and assert representations through t=63 are equal within `1e-6`.

- [ ] **Step 2: Implement causal convolution blocks**

Each block uses left padding only, Conv1d, GELU, LayerNorm over channels, residual projection when dimensions change. Channels are 32, 48, 64 with dilations 1, 2, 4.

- [ ] **Step 3: Implement history GRU with packed valid lengths**

For zero history length, return a learned `no_history` embedding. For positive length, use only unmasked summaries.

- [ ] **Step 4: Implement gated fusion**

Return both fused vector and gate tensor so later evaluation can analyze reliance on fast versus slow information.

- [ ] **Step 5: Implement complete model**

Forward signature:

```python
def forward(
    self,
    fast_x: torch.Tensor,
    fast_mask: torch.Tensor,
    slow_x: torch.Tensor,
    slow_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
```

Return:

```python
{
  "soc_q": ...,          # [B,3]
  "soh_q": ...,          # [B,3]
  "log_tte_q": ...,      # [B,3]
  "fusion_gate": ...,    # [B,64]
}
```

- [ ] **Step 6: Assert parameter count**

`sum(p.numel() for p in model.parameters()) < 500_000`.

- [ ] **Step 7: Run and commit**

```bash
python -m pytest tests/test_model.py -v
git add src/tristatelite/models configs/model/tristatelite.yaml tests/test_model.py
git commit -m "feat: implement lightweight dual-timescale TriStateLite"
```

---

### Task 10: Training Engine, Checkpointing, and One-Batch Overfit Test

**Files:**
- Create: `src/tristatelite/training/engine.py`
- Create: `src/tristatelite/training/checkpoint.py`
- Create: `src/tristatelite/config.py`
- Create: `configs/train/default.yaml`
- Create: `scripts/train.py`
- Create: `tests/test_smoke_training.py`

**Interfaces:**
- Produces: `compute_multitask_loss(outputs: dict, batch: dict, consistency_weight: float) -> tuple[Tensor, dict]`
- Produces: `train_epoch(...) -> dict[str, float]`
- Produces: `evaluate_epoch(...) -> dict[str, float]`
- Produces checkpoint with model, optimizer, config, scaler artifact, split ID, git commit and epoch

- [ ] **Step 1: Write loss-composition tests**

Assert total loss equals SOC + SOH + TTE + 0.1 consistency using fixed tensors. Assert consistency can be disabled with weight 0.

- [ ] **Step 2: Implement config loading and validation**

Reject:

- bidirectional model settings;
- window length below 32;
- non-positive stride;
- consistency weight outside `[0,1]`;
- model parameter count above 500k;
- missing split manifest.

- [ ] **Step 3: Implement training loop**

Defaults:

```yaml
seed: 0
batch_size: 64
max_epochs: 50
learning_rate: 0.001
weight_decay: 0.0001
grad_clip_norm: 1.0
early_stopping_patience: 8
num_workers: 0
consistency_weight: 0.1
```

Use AdamW, no mixed precision, validation total pinball loss for early stopping, save `best.pt` and `last.pt`.

- [ ] **Step 4: Implement one-batch overfit test**

Generate 32 synthetic samples with deterministic targets. Train 150 gradient steps. Assert final total loss is less than 40% of initial loss and all quantiles remain ordered.

- [ ] **Step 5: Implement CLI**

Required command:

```bash
python scripts/train.py \
  --data-config configs/data/nasa_randomized.yaml \
  --model-config configs/model/tristatelite.yaml \
  --train-config configs/train/default.yaml \
  --split-manifest data/manifests/split_seed_2026.json \
  --seed 0 \
  --output outputs/nasa/tristatelite/seed_0
```

- [ ] **Step 6: Run and commit**

```bash
python -m pytest tests/test_smoke_training.py -v
git add src/tristatelite/training src/tristatelite/config.py configs/train scripts/train.py tests
git commit -m "feat: add reproducible multitask training engine"
```

---

### Task 11: Evaluation, Calibration, Runtime, and Paper-Ready Reports

**Files:**
- Create: `src/tristatelite/evaluation/metrics.py`
- Create: `src/tristatelite/evaluation/calibration.py`
- Create: `src/tristatelite/evaluation/latency.py`
- Create: `src/tristatelite/evaluation/reports.py`
- Create: `scripts/evaluate.py`
- Create: `tests/test_metrics.py`

**Interfaces:**
- Produces: `regression_metrics(y_true, y_pred) -> dict[str, float]`
- Produces: `interval_metrics(y_true, q05, q95) -> dict[str, float]`
- Produces: `evaluate_checkpoint(...) -> dict[str, object]`
- Produces files: `metrics.json`, `predictions.parquet`, `summary.csv`, `calibration.csv`, `latency.json`

- [ ] **Step 1: Write exact metric tests**

Use small arrays with hand-computed MAE, RMSE, coverage and interval width. Test that restricted MAPE excludes TTE < 60 s.

- [ ] **Step 2: Implement task metrics**

SOC: MAE, RMSE, MaxAE.

SOH: MAE, RMSE, R².

TTE: MAE, RMSE, sMAPE, MAPE for target ≥60 s.

Probability: q05–q95 coverage, mean interval width, normalized interval width, pinball loss.

- [ ] **Step 3: Implement grouped reports**

Group by:

- battery ID;
- source load-condition group if available;
- early/mid/late aging thirds by cycle index;
- TTE buckets `[0,60)`, `[60,300)`, `[300,900)`, `[900,∞)` seconds.

- [ ] **Step 4: Implement latency profiling**

Warm up 20 passes, time 200 single-window passes with synchronization where supported. Report median, p90 and p99 milliseconds, device type, parameter count and serialized model size.

- [ ] **Step 5: Implement evaluation CLI**

```bash
python scripts/evaluate.py \
  --checkpoint outputs/nasa/tristatelite/seed_0/best.pt \
  --split test \
  --output outputs/nasa/tristatelite/seed_0/evaluation
```

- [ ] **Step 6: Run and commit**

```bash
python -m pytest tests/test_metrics.py -v
git add src/tristatelite/evaluation scripts/evaluate.py tests/test_metrics.py
git commit -m "feat: add calibrated evaluation and runtime reports"
```

---

### Task 12: End-to-End Dataset Build and Smoke Run

**Files:**
- Create: `scripts/build_dataset.py`
- Create: `scripts/run_smoke.sh`
- Create: `README.md`
- Update: `Makefile`

**Interfaces:**
- Produces: canonical parquet files, cycle summaries, split manifest, scaler artifact and audit reports

- [ ] **Step 1: Implement build pipeline CLI**

```bash
python scripts/build_dataset.py \
  --config configs/data/nasa_randomized.yaml \
  --raw data/raw/battery_alt_dataset.zip \
  --output data/processed/nasa_randomized \
  --split-seed 2026 \
  --protocol calibrated
```

Execution order must be inventory check → adapter → canonicalization → cycle validation → labels → split → leakage audit → train scaler → processed files.

- [ ] **Step 2: Add build success report**

`data/processed/nasa_randomized/build_report.json` must contain:

- archive SHA256;
- battery count;
- accepted/rejected cycle counts;
- sample counts by split;
- missing temperature rate;
- SOC/SOH/TTE ranges;
- split overlap audit;
- config hash;
- build timestamp.

- [ ] **Step 3: Add `--limit-batteries` and `--limit-cycles` only for smoke tests**

The generated artifact must record that limits were active, and production training must reject limited artifacts unless `--allow-smoke-data` is explicitly passed.

- [ ] **Step 4: Create the smoke script**

```bash
#!/usr/bin/env bash
set -euo pipefail
python scripts/build_dataset.py \
  --config configs/data/nasa_randomized.yaml \
  --raw data/raw/battery_alt_dataset.zip \
  --output data/processed/nasa_smoke \
  --split-seed 2026 \
  --protocol calibrated \
  --limit-batteries 6 \
  --limit-cycles 8
python scripts/train.py \
  --data-config configs/data/nasa_randomized.yaml \
  --model-config configs/model/tristatelite.yaml \
  --train-config configs/train/default.yaml \
  --split-manifest data/processed/nasa_smoke/split_manifest.json \
  --seed 0 \
  --max-epochs 2 \
  --allow-smoke-data \
  --output outputs/smoke
python scripts/evaluate.py \
  --checkpoint outputs/smoke/best.pt \
  --split test \
  --output outputs/smoke/evaluation
```

- [ ] **Step 5: Run all tests and smoke pipeline**

```bash
python -m pytest -q
bash scripts/run_smoke.sh
```

Expected:

- all tests pass;
- `outputs/smoke/best.pt` exists;
- `outputs/smoke/evaluation/metrics.json` exists;
- no leakage audit failure;
- model parameter count <500k.

- [ ] **Step 6: Commit**

```bash
git add scripts/build_dataset.py scripts/run_smoke.sh README.md Makefile
git commit -m "feat: complete NASA-to-TriStateLite smoke pipeline"
```

---

### Task 13: Full NASA Baselines, Ablations, and Three-Seed Run

**Files:**
- Create: `scripts/run_experiments.py`
- Create: `configs/experiments/nasa_main.yaml`
- Extend: `src/tristatelite/evaluation/reports.py`

**Interfaces:**
- Produces experiment registry and aggregate `summary_mean_std.csv`

- [ ] **Step 1: Define experiment matrix**

```yaml
seeds: [0, 1, 2]
models:
  - single_gru_soc
  - single_gru_soh
  - single_gru_tte
  - shared_gru
  - fast_tcn
  - tristatelite
ablations:
  - tristatelite_no_history
  - tristatelite_no_consistency
  - tristatelite_concat_fusion
protocols: [calibrated, zero_shot]
```

- [ ] **Step 2: Implement resumable experiment runner**

Each run writes `status.json` with config hash. Skip only runs whose completed hash matches. Failed runs must preserve stderr and traceback.

- [ ] **Step 3: Execute one full seed before the matrix**

Run only `tristatelite`, seed 0. Review learning curves, prediction ranges and per-battery metrics. Stop if:

- validation loss is NaN;
- median predictions are constant;
- q90 coverage is below 20% or above 99.9%;
- test metrics are implausibly much better than train/validation without explanation;
- any leakage audit warning occurs.

- [ ] **Step 4: Run all three seeds and aggregate**

```bash
python scripts/run_experiments.py --config configs/experiments/nasa_main.yaml
```

- [ ] **Step 5: Commit experiment definitions, not large outputs**

```bash
git add scripts/run_experiments.py configs/experiments src/tristatelite/evaluation/reports.py
git commit -m "exp: define NASA baselines ablations and three-seed study"
```

---

### Task 14: CALCE External-Validation Adapter

**Files:**
- Create: `src/tristatelite/data/calce_adapter.py`
- Create: `configs/data/calce.yaml`
- Create: `tests/test_calce_adapter.py`
- Create: `configs/experiments/calce_external.yaml`

**Interfaces:**
- Produces canonical sample table identical to NASA adapter output
- Reuses labels, split, scaling, windowing, model and evaluation code unchanged

- [ ] **Step 1: Inventory downloaded CALCE files before semantic parsing**

Record filenames, workbook sheet names, column headers and cell IDs. Select a coherent subset with time, voltage, current and repeated cycles; do not mix chemistries in the first external test.

- [ ] **Step 2: Write adapter tests using a synthetic XLSX fixture**

Assert canonical columns, current sign, battery/cycle IDs and causal timestamps.

- [ ] **Step 3: Implement the CALCE adapter only**

Do not duplicate label or window logic. The adapter ends at the canonical schema boundary.

- [ ] **Step 4: Run frozen-model external evaluation**

First evaluate the NASA-trained model without CALCE fine-tuning. Then run a separately labeled fine-tuning experiment. Keep zero-shot and adapted results separate.

- [ ] **Step 5: Commit**

```bash
git add src/tristatelite/data/calce_adapter.py configs/data/calce.yaml \
  configs/experiments/calce_external.yaml tests/test_calce_adapter.py
git commit -m "feat: add CALCE external validation adapter"
```

---

## Immediate Execution Order for Codex

Codex should execute Tasks 1–2 first and stop after producing:

```text
data/raw/battery_alt_dataset.zip
data/manifests/nasa_archive_inventory.json
```

A human must review the real archive structure before Task 3. After the mapping is confirmed, Codex executes Tasks 3–12. Task 13 starts only after the smoke pipeline passes. Task 14 starts only after the NASA three-seed result is stable.

## Completion Verification

Before claiming the MVP is complete, run:

```bash
python -m pytest -q
ruff check src tests scripts
bash scripts/run_smoke.sh
python scripts/train.py \
  --data-config configs/data/nasa_randomized.yaml \
  --model-config configs/model/tristatelite.yaml \
  --train-config configs/train/default.yaml \
  --split-manifest data/processed/nasa_randomized/split_manifest.json \
  --seed 0 \
  --output outputs/nasa/tristatelite/seed_0
python scripts/evaluate.py \
  --checkpoint outputs/nasa/tristatelite/seed_0/best.pt \
  --split test \
  --output outputs/nasa/tristatelite/seed_0/evaluation
```

The completion report must state the actual:

- accepted battery and cycle counts;
- train/validation/test battery IDs;
- archive and config hashes;
- parameter count;
- device used;
- best validation epoch;
- SOC, SOH and TTE test metrics;
- 90% interval coverage and width;
- median/p90/p99 inference latency;
- unresolved data-quality limitations.
