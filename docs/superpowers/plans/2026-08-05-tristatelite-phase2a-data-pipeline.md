# TriStateLite Phase 2A Leakage-Safe Data Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the audited NASA accelerated-life CSV archive into reproducible, causal, battery-isolated SOC/SOH/TTE training windows without implementing a model.

**Architecture:** A YAML-backed NASA source adapter streams only physical `batteryNN.csv` members and assigns battery metadata from validated paths. A cycle builder identifies maximal contiguous `mode == -1` runs, causally resamples them to 1 Hz, validates completeness, integrates capacity, and generates labels. Battery-level splitting precedes train-only scaling and causal window caching; a CLI writes partitioned Parquet artifacts plus provenance and leakage reports.

**Tech Stack:** Python 3.11, NumPy, pandas, PyArrow, PyYAML, PyTorch, pytest, ruff, nbformat/nbclient for the reproducible source audit.

## Global Constraints

- Source archive SHA256 must equal `89209acddbad47506d781698fc51ebe9d7cdb8968667698f63e63aa730981d21`.
- Use only members matching `battery_alt_dataset/*/battery[0-9][0-9].csv`; never ingest `__MACOSX` members.
- README-confirmed units are seconds, volts, amperes, and degrees Celsius.
- README-confirmed modes are `-1=discharge`, `0=rest`, `1=charge`; `mission_type=0` is reference discharge and `1` is regular mission.
- A cycle is one maximal contiguous `mode == -1` run; `start_time` is metadata, not a cycle key.
- Current is already positive during discharge; reject negative integrated discharge capacity rather than silently flipping signs.
- Resample causally at 1 Hz by flooring timestamps, taking the last observation per second, and forward-filling at most 5 seconds. Never use future interpolation.
- Reject cycles with raw duration below 60 seconds, fewer than 50 raw samples, non-positive capacity, non-positive net voltage drop after ignoring load-board sentinel voltage ≤1 V, or a remaining voltage/current gap.
- Split by battery ID only. Fit all imputers/scalers on training batteries only.
- Current-cycle final capacity may generate labels but may not enter model features, scaling statistics, or slow history for that cycle.
- Cache identity must include archive hash, split ID, preprocessing config hash, window length, and stride.
- Phase 2A stops after dataset/window smoke verification. Do not create model, loss, training, or evaluation modules.

---

## File Structure

```text
configs/data/nasa_randomized.yaml       # audited source mapping and preprocessing rules
docs/data_dictionary.md                 # canonical fields, units, and label definitions
docs/leakage_contract.md                # enforceable leakage rules
scripts/build_dataset.py                # Phase 2A orchestration CLI
src/tristatelite/config.py              # YAML loading and stable config hashing
src/tristatelite/schemas.py             # immutable canonical/split contracts
src/tristatelite/data/nasa_adapter.py   # safe ZIP member discovery and streaming
src/tristatelite/data/canonicalize.py   # discharge runs and causal 1 Hz samples
src/tristatelite/data/labels.py         # capacity, targets, and cycle summaries
src/tristatelite/data/split.py          # deterministic battery-level split
src/tristatelite/data/scaling.py        # train-only robust scaler
src/tristatelite/data/audit.py          # leakage and artifact gates
src/tristatelite/data/windows.py        # causal fast/slow window dataset and cache ID
tests/fixtures/synthetic_cycle.py       # deterministic raw-cycle fixtures
tests/test_config.py
tests/test_nasa_adapter.py
tests/test_canonicalize.py
tests/test_labels.py
tests/test_split.py
tests/test_scaling.py
tests/test_windows.py
tests/test_build_dataset.py
```

### Task 1: Audited Configuration and Canonical Contracts

**Files:**
- Create: `configs/data/nasa_randomized.yaml`
- Create: `src/tristatelite/config.py`
- Create: `src/tristatelite/schemas.py`
- Create: `tests/test_config.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Produces: `load_yaml(path: Path) -> dict[str, object]`
- Produces: `config_hash(config: Mapping[str, object]) -> str`
- Produces: `CanonicalSample`, `CycleSummary`, and `SplitManifest` frozen dataclasses.

- [ ] **Step 1: Write failing configuration tests**

```python
def test_nasa_config_matches_audited_schema():
    config = load_yaml(Path("configs/data/nasa_randomized.yaml"))
    assert config["archive_sha256"] == "89209acddbad47506d781698fc51ebe9d7cdb8968667698f63e63aa730981d21"
    assert config["member_pattern"] == "battery_alt_dataset/*/battery[0-9][0-9].csv"
    assert config["fields"]["time_s"] == "time"
    assert config["fields"]["current_a"] == "current_load"
    assert config["mode_map"] == {-1: "discharge", 0: "rest", 1: "charge"}

def test_config_hash_is_order_independent():
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})
```

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/python -m pytest tests/test_config.py -v`

Expected: import failure for `tristatelite.config` and `tristatelite.schemas`.

- [ ] **Step 3: Implement exact audited configuration**

```yaml
archive_sha256: 89209acddbad47506d781698fc51ebe9d7cdb8968667698f63e63aa730981d21
member_pattern: battery_alt_dataset/*/battery[0-9][0-9].csv
battery_id_source: filename_stem
group_source: parent_directory
fields:
  start_time: start_time
  time_s: time
  mode: mode
  voltage_v: voltage_load
  current_a: current_load
  temperature_c: temperature_battery
  mission_type: mission_type
mode_map: {-1: discharge, 0: rest, 1: charge}
cycle_rule: maximal_contiguous_discharge_mode
resample_hz: 1
max_forward_fill_s: 5
minimum_duration_s: 60
minimum_samples: 50
minimum_valid_voltage_v: 1.0
current_sign: discharge_positive
```

- [ ] **Step 4: Implement stable JSON hashing and frozen dataclasses**

`config_hash()` must serialize with `sort_keys=True`, compact separators, UTF-8, and SHA256. `SplitManifest` stores tuples of battery IDs and the config/archive hashes used to create it.

- [ ] **Step 5: Verify GREEN and commit**

Run: `.venv/bin/python -m pytest tests/test_config.py -v && .venv/bin/ruff check src tests scripts`

Commit: `git commit -m "feat: define audited NASA canonical contracts"`

### Task 2: Safe NASA ZIP Adapter and Discharge-Run Discovery

**Files:**
- Create: `src/tristatelite/data/nasa_adapter.py`
- Create: `tests/fixtures/synthetic_cycle.py`
- Create: `tests/test_nasa_adapter.py`

**Interfaces:**
- Produces: `discover_battery_members(archive: Path, config: Mapping[str, object]) -> list[BatteryMember]`
- Produces: `iter_battery_chunks(archive: Path, member: BatteryMember, chunksize: int = 250_000) -> Iterator[pd.DataFrame]`
- Produces: `iter_discharge_runs(chunks: Iterable[pd.DataFrame]) -> Iterator[pd.DataFrame]`

- [ ] **Step 1: Write failing member-discovery tests**

Create a synthetic ZIP with one physical member, one README, and one `__MACOSX` resource member. Assert only the physical CSV is returned, `battery_id="battery00"`, group is the parent directory, paths are sorted, and archive hash mismatch raises `ValueError` before reading CSV data.

- [ ] **Step 2: Write failing cross-chunk run tests**

Use chunks where a discharge begins in chunk one and ends in chunk two. Assert `iter_discharge_runs()` yields exactly one combined run, then a later discharge as a second run. Assert `start_time` changes inside a run do not create a new cycle, while a mode transition does.

- [ ] **Step 3: Verify RED**

Run: `.venv/bin/python -m pytest tests/test_nasa_adapter.py -v`

- [ ] **Step 4: Implement safe discovery and streaming**

Reject unsafe/encrypted members, require the exact ten audited columns, coerce numeric columns with `errors="coerce"`, treat stripped case-insensitive `NAN` as missing, and raise on unsupported mode values. `iter_discharge_runs()` buffers only the current discharge run and yields a copy when mode exits `-1`.

- [ ] **Step 5: Verify GREEN and commit**

Run: `.venv/bin/python -m pytest tests/test_nasa_adapter.py -v`

Commit: `git commit -m "feat: stream audited NASA discharge runs"`

### Task 3: Causal 1 Hz Canonicalization and Cycle Validation

**Files:**
- Create: `src/tristatelite/data/canonicalize.py`
- Create: `tests/test_canonicalize.py`

**Interfaces:**
- Consumes: raw discharge runs from Task 2.
- Produces: `canonicalize_discharge(run: pd.DataFrame, battery_id: str, cycle_index: int, config: Mapping[str, object]) -> tuple[pd.DataFrame | None, dict[str, object]]`
- Produces canonical columns: `battery_id`, `cycle_id`, `cycle_index`, `timestamp_s`, `elapsed_s`, `voltage_v`, `current_a`, `temperature_c`, `temperature_available`, `mission_type`, `phase`.

- [ ] **Step 1: Write failing causal-resampling tests**

Use timestamps `[0.2, 0.8, 2.1, 4.2]`. Assert second 0 keeps the last observation, seconds 1 and 3 forward-fill only from the past, no timestamp precedes the first observed second, and output timestamps are strictly increasing.

- [ ] **Step 2: Write failing validity tests**

Assert rejection reason codes for `too_short`, `too_few_samples`, `non_positive_capacity`, `non_decreasing_voltage`, `missing_required_signal`, and `gap_exceeds_limit`. Assert the load-board sentinel `-0.026 V` is ignored when selecting start/end voltage for the trend check.

- [ ] **Step 3: Verify RED**

Run: `.venv/bin/python -m pytest tests/test_canonicalize.py -v`

- [ ] **Step 4: Implement causal canonicalization**

Sort by `time`, reject non-increasing raw timestamps, floor lifetime time to integer seconds, keep the last row per second, reindex only from first to last observed second, forward-fill voltage/current/temperature/mission for at most five seconds, and preserve temperature availability before filling. Do not fill temperature with a global value in this task.

- [ ] **Step 5: Verify GREEN and commit**

Run: `.venv/bin/python -m pytest tests/test_canonicalize.py -v`

Commit: `git commit -m "feat: canonicalize and validate NASA discharges"`

### Task 4: Capacity, SOC/SOH/TTE Labels, and Historical Summaries

**Files:**
- Create: `src/tristatelite/data/labels.py`
- Create: `tests/test_labels.py`
- Create: `docs/data_dictionary.md`

**Interfaces:**
- Produces: `integrate_discharge_ah(time_s: np.ndarray, current_a: np.ndarray) -> np.ndarray`
- Produces: `label_battery_cycles(cycles: list[pd.DataFrame], protocol: str, global_q_ref_ah: float | None = None) -> tuple[list[pd.DataFrame], pd.DataFrame]`

- [ ] **Step 1: Write failing numeric integration tests**

Assert one ampere for 3,600 seconds integrates to exactly 1 Ah within `1e-4`, cumulative capacity starts at zero, negative current raises, and time reversal raises.

- [ ] **Step 2: Write failing label tests**

For three deterministic discharges, assert SOC begins at 1 and ends at 0, TTE ends at 0 and never increases, `log_tte_target == log1p(tte_seconds)`, SOH is constant per cycle, and calibrated Q-ref is the median final capacity of the earliest three valid cycles.

- [ ] **Step 3: Write leakage-history tests**

Assert cycle-summary row `k` may be used only by cycles with index greater than `k`; no summary field named `current_cycle_capacity`, `target`, `future`, or `final_capacity` is emitted as a model feature.

- [ ] **Step 4: Implement labels and summaries**

Use cumulative trapezoidal Ah. Add `soc_target`, `soh_target`, `tte_seconds`, `log_tte_target`, and `q_ref_ah`. Summaries contain delivered Ah, duration, mean voltage/current/temperature, current standard deviation, temperature rise, and voltage-slope mean. Log unclipped label ranges and fail if more than 1% lie outside `[-0.02, 1.05]` before clipping.

- [ ] **Step 5: Verify GREEN and commit**

Run: `.venv/bin/python -m pytest tests/test_labels.py -v`

Commit: `git commit -m "feat: generate leakage-safe battery targets"`

### Task 5: Battery Split, Train-Only Scaling, and Leakage Audit

**Files:**
- Create: `src/tristatelite/data/split.py`
- Create: `src/tristatelite/data/scaling.py`
- Create: `src/tristatelite/data/audit.py`
- Create: `tests/test_split.py`
- Create: `tests/test_scaling.py`
- Create: `docs/leakage_contract.md`

**Interfaces:**
- Produces: `make_group_split(battery_ids: list[str], seed: int = 2026, fractions: tuple[float, float, float] = (0.7, 0.15, 0.15)) -> SplitManifest`
- Produces: `fit_scaler(train: pd.DataFrame, features: list[str]) -> dict[str, object]`
- Produces: `transform_features(frame: pd.DataFrame, artifact: Mapping[str, object]) -> pd.DataFrame`
- Produces: `audit_leakage(manifest: SplitManifest, samples: pd.DataFrame, history: pd.DataFrame, scaler_fit_batteries: set[str]) -> dict[str, object]`

- [ ] **Step 1: Write failing deterministic split tests**

Use the 26 audited IDs. Assert repeatability, pairwise disjoint sets, complete union, non-empty splits, and approximate sizes 18/4/4. A sample inherits split solely from `battery_id`.

- [ ] **Step 2: Write failing train-only scaling tests**

Fit on values centered at zero and transform held-out values centered at 100. Assert stored median/IQR equal train statistics, held-out transformed values remain shifted, unavailable temperature fills with the training median, and `temperature_available` is unchanged.

- [ ] **Step 3: Write failing leakage-audit tests**

Assert failures for battery overlap, unassigned batteries, scaler fit on held-out batteries, history cycle index not less than current index, forbidden feature names, and duplicate sample keys across splits.

- [ ] **Step 4: Implement split/scaler/audit**

Sort IDs before a local NumPy permutation. Scale with `(x - median) / max(IQR, 1e-6)` and clip to `[-10, 10]`. Return a JSON-serializable audit with zero-overlap evidence and explicit fitted battery IDs.

- [ ] **Step 5: Verify GREEN and commit**

Run: `.venv/bin/python -m pytest tests/test_split.py tests/test_scaling.py -v`

Commit: `git commit -m "feat: enforce battery isolation and train-only scaling"`

### Task 6: Causal Fast/Slow Windows and Provenance Cache Identity

**Files:**
- Create: `src/tristatelite/data/windows.py`
- Create: `tests/test_windows.py`

**Interfaces:**
- Produces: `engineer_causal_features(samples: pd.DataFrame) -> pd.DataFrame`
- Produces: `BatteryWindowDataset(samples, cycle_summaries, battery_ids, feature_names, fast_length=128, history_length=8, stride=10)`
- Produces: `cache_identity(archive_sha256: str, split_id: str, config_hash: str, fast_length: int, stride: int) -> str`

- [ ] **Step 1: Write failing causal-feature invariance tests**

Create two series identical through second 100 and different afterward. Assert every engineered feature through second 100 is identical. Cover deltas, current mean/std at 10 seconds, current mean/CV at 60 seconds, voltage slope at 30 seconds, and temperature slope at 60 seconds.

- [ ] **Step 2: Write failing window/history tests**

Assert early windows left-pad with zeros and return `fast_mask`; slow history contains only completed cycles with lower cycle indices, left-pads to 8, and returns `slow_mask`; metadata retains battery/cycle/time; training stride 10 and evaluation stride 1 yield expected lengths.

- [ ] **Step 3: Write failing cache tests**

Assert any change to archive hash, split ID, config hash, fast length, or stride changes the cache identity.

- [ ] **Step 4: Implement features and dataset**

Use backward-only pandas rolling windows and one-sided slopes. Dataset items contain `fast_x`, `fast_mask`, `slow_x`, `slow_mask`, `targets`, `physics`, and `metadata`. Never expose target or Q-end columns in `fast_x` or `slow_x`.

- [ ] **Step 5: Verify GREEN and commit**

Run: `.venv/bin/python -m pytest tests/test_windows.py -v`

Commit: `git commit -m "feat: build causal dual-timescale windows"`

### Task 7: Partitioned Dataset Build CLI and Real Smoke Gate

**Files:**
- Create: `scripts/build_dataset.py`
- Create: `tests/test_build_dataset.py`
- Modify: `Makefile`
- Modify: `README.md`

**Interfaces:**
- CLI: `python scripts/build_dataset.py --config configs/data/nasa_randomized.yaml --archive data/raw/battery_alt_dataset.zip --output data/processed/nasa_randomized --split-seed 2026 --protocol calibrated [--limit-batteries N] [--limit-cycles N]`
- Produces partitioned samples, `cycle_summaries.parquet`, `split_manifest.json`, `scaler.json`, `rejected_cycles.csv`, `leakage_audit.json`, and `build_report.json`.

- [ ] **Step 1: Write failing synthetic end-to-end test**

Build a ZIP with six small batteries and at least three valid discharges each. Run the CLI with limits. Assert every artifact exists, processed sample keys are unique, battery splits do not overlap, rejected cycles carry reason codes, scaler fit IDs equal training IDs, and the build report records `limited=true`.

- [ ] **Step 2: Implement streaming orchestration**

Execute in this order: hash/config validation → member discovery → discharge runs → canonicalization/rejection → labels/summaries → grouped split → train scaler → transformation → causal features → partitioned Parquet write → leakage audit → report. Write to a temporary output directory and atomically rename only after all gates pass.

- [ ] **Step 3: Add provenance report**

`build_report.json` records archive/config hashes, battery and accepted/rejected cycle counts, samples by split, temperature unavailable/invalid rates, target ranges, split overlap, protocol, limits, and UTC build time.

- [ ] **Step 4: Run the synthetic gate**

Run: `.venv/bin/python -m pytest tests/test_build_dataset.py -v`

- [ ] **Step 5: Run a real six-battery smoke build**

```bash
.venv/bin/python scripts/build_dataset.py \
  --config configs/data/nasa_randomized.yaml \
  --archive data/raw/battery_alt_dataset.zip \
  --output data/processed/nasa_smoke \
  --split-seed 2026 \
  --protocol calibrated \
  --limit-batteries 6 \
  --limit-cycles 12
```

Assert at least six batteries survive, every battery has at least three accepted cycles, all leakage checks pass, and no limited artifact is accepted as production input without an explicit future `--allow-smoke-data` flag.

- [ ] **Step 6: Run the Phase 2A completion gate**

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check src tests scripts
git status --short
```

Also assert no files exist under `src/tristatelite/models`, `losses`, `training`, or `evaluation`.

- [ ] **Step 7: Commit and stop**

Commit: `git commit -m "feat: complete leakage-safe NASA data pipeline"`

Stop before any model or training implementation. Phase 2B begins only after the real smoke build report is reviewed.

## Plan Self-Review

- Spec coverage: audited mapping, member exclusion, cycle semantics, causal resampling, capacity/labels, grouped split, train-only scaling, leakage audit, windows, provenance, and real smoke build each map to an explicit task.
- Placeholder scan: every implementation step has concrete behavior, commands, and expected evidence.
- Type consistency: Task 2 yields raw runs consumed by Task 3; Task 3 canonical frames feed Task 4; Task 4 labels and summaries feed Tasks 5–7; cache and split identifiers use the hashes introduced in Task 1.
- Scope: model, loss, training, evaluation, baselines, and CALCE are explicitly excluded from Phase 2A.
