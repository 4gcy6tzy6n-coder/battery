# TriStateLite Phase 2B Probabilistic Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve unit-bearing physics channels and implement ordered SOC/SOH/log-TTE quantile heads, pinball loss, and load-conditioned physics consistency as independently tested primitives.

**Architecture:** Causal features are computed in physical units before any fitted transformation. Robust scaling writes separate `__scaled` columns for model inputs, while window items expose a named unscaled physics mapping. Small PyTorch modules then enforce quantile ordering by construction and compute strictly validated statistical and physical losses.

**Tech Stack:** Python 3.11, PyTorch 2.x, NumPy, pandas, PyArrow, pytest, Ruff.

## Global Constraints

- Raw physical columns retain their documented units in processed Parquet artifacts.
- Fitted transformations use training batteries only.
- Every engineered feature is causal and isolated by battery and cycle.
- Model inputs and physics inputs use distinct, explicit column names.
- Quantile order is fixed at `(0.05, 0.50, 0.95)` and is never produced by sorting outputs.
- The physics loss is active only when finite `i_eff_60s > 0.05 A`.
- Physical TTE is clamped only inside the consistency loss to `[0, 604800]` seconds.
- Baseline models, complete TriStateLite, training, and evaluation remain out of scope.

---

## File Structure

```text
docs/data_dictionary.md                      # unit-bearing and scaled field contract
scripts/build_dataset.py                     # raw feature -> train scaler -> scaled copy order
src/tristatelite/data/audit.py               # model-feature name leakage gate
src/tristatelite/data/scaling.py             # optional suffixed transformation
src/tristatelite/data/windows.py             # physical causal fields and item contract
src/tristatelite/models/__init__.py           # model package
src/tristatelite/models/quantile_heads.py     # ordered bounded and positive heads
src/tristatelite/losses/__init__.py           # loss package
src/tristatelite/losses/pinball.py            # validated quantile regression loss
src/tristatelite/losses/consistency.py        # physical TTE consistency loss
tests/test_scaling.py                         # raw preservation and scaled copies
tests/test_windows.py                         # causal physical fields and dataset mapping
tests/test_build_dataset.py                   # artifact-level unit separation
tests/test_quantile_heads.py                  # ordering, bounds, gradients, shapes
tests/test_losses.py                          # numeric loss and physical diagnostics
```

### Task 1: Preserve Physical Features and Add Suffixed Scaling

**Files:**
- Modify: `src/tristatelite/data/windows.py`
- Modify: `src/tristatelite/data/scaling.py`
- Modify: `tests/test_windows.py`
- Modify: `tests/test_scaling.py`

**Interfaces:**
- Produces: `MODEL_CONTINUOUS_FEATURES: tuple[str, ...]`
- Produces: `MODEL_FEATURES: tuple[str, ...]`
- Produces: `engineer_causal_features(samples: pd.DataFrame) -> pd.DataFrame`
- Produces: `transform_features(frame: pd.DataFrame, artifact: Mapping[str, object], *, output_suffix: str | None = None) -> pd.DataFrame`

^- [x] **Step 1: Write failing physical-feature tests**

Extend `tests/test_windows.py` so the future-invariance test includes the exact
columns below and verifies their units on a constant-current sequence:

```python
engineered = [
    "voltage_delta_1s",
    "current_delta_1s",
    "temperature_delta_1s",
    "current_mean_10s",
    "current_std_10s",
    "i_eff_60s",
    "current_cv_60s",
    "voltage_slope_30s",
    "temperature_slope_60s",
]

features = engineer_causal_features(_series(0.0))
assert features.loc[100, "i_eff_60s"] == pytest.approx(_series(0.0).loc[41:100, "current_a"].mean())
assert np.isfinite(features["current_cv_60s"]).all()
```

Add a sequence with exactly `2.5 A` and assert `i_eff_60s == 2.5` and
`current_cv_60s == 0` after the first full window.

^- [x] **Step 2: Write a failing suffixed-scaler test**

Extend `tests/test_scaling.py`:

```python
transformed = transform_features(held_out, artifact, output_suffix="__scaled")
assert transformed["voltage_v"].tolist() == [100.0, 101.0]
assert transformed["voltage_v__scaled"].tolist() == [10.0, 10.0]
assert transformed["temperature_c"].isna().iloc[0]
assert transformed["temperature_c__scaled"].iloc[0] == 0.0
assert transformed["temperature_available"].tolist() == [False, True]
```

^- [x] **Step 3: Verify RED**

Run:

```bash
.venv/bin/pytest tests/test_windows.py tests/test_scaling.py -v
```

Expected: failures for missing `temperature_delta_1s`, missing `i_eff_60s`, and
unsupported `output_suffix`.

^- [x] **Step 4: Implement causal physical features**

In `src/tristatelite/data/windows.py`, compute all rolling fields before scaling:

```python
result["temperature_delta_1s"] = result["temperature_c"].diff().fillna(0.0)
result["i_eff_60s"] = result["current_a"].rolling(60, min_periods=1).mean()
current_std_60s = result["current_a"].rolling(60, min_periods=1).std(ddof=0)
result["current_cv_60s"] = (
    current_std_60s / (result["i_eff_60s"].abs() + 1e-3)
).fillna(0.0)
```

Define:

```python
MODEL_CONTINUOUS_FEATURES = (
    "voltage_v",
    "current_a",
    "temperature_c",
    "voltage_delta_1s",
    "current_delta_1s",
    "temperature_delta_1s",
    "current_mean_10s",
    "current_std_10s",
    "i_eff_60s",
    "voltage_slope_30s",
    "temperature_slope_60s",
)
MODEL_FEATURES = tuple(f"{name}__scaled" for name in MODEL_CONTINUOUS_FEATURES) + (
    "temperature_available",
)
```

Remove `current_mean_60s`; `i_eff_60s` is its explicit unit-bearing replacement.

^- [x] **Step 5: Implement suffix-preserving transformation**

Change `transform_features` so `output_suffix=None` retains the existing overwrite
behavior and any string suffix writes a new column:

```python
destination = str(feature) if output_suffix is None else f"{feature}{output_suffix}"
result[destination] = np.clip((values - median) / iqr, lower, upper)
```

Reject a suffix that would overwrite an existing source name and reject missing
source columns with a clear `ValueError`.

^- [x] **Step 6: Verify GREEN and commit**

Run:

```bash
.venv/bin/pytest tests/test_windows.py tests/test_scaling.py -v
.venv/bin/ruff check src/tristatelite/data tests/test_windows.py tests/test_scaling.py
```

Commit:

```bash
git add src/tristatelite/data/windows.py src/tristatelite/data/scaling.py tests/test_windows.py tests/test_scaling.py
git commit -m "fix: preserve unit-bearing physics features"
```

### Task 2: Enforce the Processed Artifact and Window Physics Contract

**Files:**
- Modify: `scripts/build_dataset.py`
- Modify: `src/tristatelite/data/audit.py`
- Modify: `src/tristatelite/data/windows.py`
- Modify: `tests/test_build_dataset.py`
- Modify: `tests/test_split.py`
- Modify: `tests/test_windows.py`
- Modify: `docs/data_dictionary.md`

**Interfaces:**
- Consumes: `MODEL_CONTINUOUS_FEATURES`, `MODEL_FEATURES`
- Extends: `audit_leakage(..., model_feature_names: list[str] | None = None) -> dict[str, object]`
- Produces each dataset item with `physics: dict[str, Tensor]`
- Produces build report keys `physical_ranges` and `scaled_feature_ranges`

^- [x] **Step 1: Write failing dataset-item tests**

Update `_window_frames()` in `tests/test_windows.py` to include:

```python
"i_eff_60s": 1.0,
"current_cv_60s": 0.0,
```

Then assert:

```python
assert set(first["physics"]) == {"i_eff_60s", "current_cv_60s", "q_ref_ah"}
assert first["physics"]["i_eff_60s"].item() == pytest.approx(1.0)
assert first["physics"]["q_ref_ah"].item() == pytest.approx(1.0)
```

Add a test that omitting any required physics field raises `ValueError` during
`BatteryWindowDataset` construction.

^- [x] **Step 2: Write failing build-contract tests**

Extend `tests/test_build_dataset.py` after reading samples:

```python
assert samples["current_a"].eq(1.0).all()
assert "current_a__scaled" in samples
assert "i_eff_60s" in samples
assert samples["i_eff_60s"].eq(1.0).all()
assert samples["current_cv_60s"].eq(0.0).all()
assert set(report["physical_ranges"]) == {"current_a", "i_eff_60s", "current_cv_60s", "q_ref_ah"}
assert "current_a__scaled" in report["scaled_feature_ranges"]
```

^- [x] **Step 3: Write a failing model-feature leakage test**

Add to `tests/test_split.py`:

```python
with pytest.raises(ValueError, match="forbidden model feature"):
    audit_leakage(
        manifest,
        samples,
        history,
        set(manifest.train_batteries),
        model_feature_names=["voltage_v__scaled", "soc_target"],
    )
```

Also verify `MODEL_FEATURES` passes the same audit.

^- [x] **Step 4: Verify RED**

Run:

```bash
.venv/bin/pytest tests/test_windows.py tests/test_build_dataset.py tests/test_split.py -v
```

Expected: positional physics tensor, overwritten raw current, absent report ranges,
and unsupported audit argument.

^- [x] **Step 5: Reorder and separate the dataset build**

In `scripts/build_dataset.py`:

1. assign splits;
2. call `engineer_causal_features(samples)` while values are raw;
3. fit the scaler on training rows and `MODEL_CONTINUOUS_FEATURES`;
4. call `transform_features(samples, scaler, output_suffix="__scaled")`;
5. audit using `model_feature_names=list(MODEL_FEATURES)`.

Generate JSON-safe ranges with finite minima and maxima:

```python
def _ranges(frame: pd.DataFrame, columns: tuple[str, ...]) -> dict[str, dict[str, float]]:
    return {
        name: {"min": float(frame[name].min()), "max": float(frame[name].max())}
        for name in columns
    }
```

Record raw ranges for `current_a`, `i_eff_60s`, `current_cv_60s`, and `q_ref_ah`,
plus ranges for every continuous `__scaled` feature.

^- [x] **Step 6: Implement named physics items and feature audit**

Validate these columns in `BatteryWindowDataset.__init__`:

```python
PHYSICS_FEATURES = ("i_eff_60s", "current_cv_60s", "q_ref_ah")
```

Return scalar float32 tensors by name. Extend `audit_leakage` with an optional
model feature list and reject names containing `target`, `future`,
`final_capacity`, `current_cycle_capacity`, or exactly `q_ref_ah`.

^- [x] **Step 7: Update the data dictionary**

Document `i_eff_60s`, `current_cv_60s`, all `__scaled` fields, the fact that raw
unit-bearing fields are never overwritten, and the named physics mapping.

^- [x] **Step 8: Verify GREEN and commit**

Run:

```bash
.venv/bin/pytest tests/test_windows.py tests/test_build_dataset.py tests/test_split.py -v
.venv/bin/ruff check src tests scripts
```

Commit:

```bash
git add scripts/build_dataset.py src/tristatelite/data tests/test_build_dataset.py tests/test_split.py tests/test_windows.py docs/data_dictionary.md
git commit -m "fix: separate model inputs from physical supervision"
```

### Task 3: Ordered Quantile Heads

**Files:**
- Create: `src/tristatelite/models/__init__.py`
- Create: `src/tristatelite/models/quantile_heads.py`
- Create: `tests/test_quantile_heads.py`

**Interfaces:**
- Produces: `QUANTILES = (0.05, 0.50, 0.95)`
- Produces: `BoundedQuantileHead(in_dim: int)`
- Produces: `PositiveQuantileHead(in_dim: int)`
- Both consume `Tensor[..., in_dim]` and return `Tensor[..., 3]`

^- [x] **Step 1: Write failing ordering and bound tests**

Create `tests/test_quantile_heads.py`:

```python
import pytest
import torch

from tristatelite.models.quantile_heads import BoundedQuantileHead, PositiveQuantileHead


@pytest.mark.parametrize("head_type", [BoundedQuantileHead, PositiveQuantileHead])
def test_quantile_heads_are_ordered_for_random_inputs(head_type):
    torch.manual_seed(7)
    output = head_type(16)(torch.randn(1000, 16))
    assert output.shape == (1000, 3)
    assert torch.all(output[:, 0] <= output[:, 1])
    assert torch.all(output[:, 1] <= output[:, 2])
    assert torch.all(output >= 0)


def test_bounded_head_stays_at_or_below_one():
    output = BoundedQuantileHead(8)(torch.randn(1000, 8) * 100)
    assert torch.all(output <= 1)


@pytest.mark.parametrize("head_type", [BoundedQuantileHead, PositiveQuantileHead])
def test_head_preserves_leading_dimensions_and_gradients(head_type):
    hidden = torch.randn(2, 5, 4, requires_grad=True)
    output = head_type(4)(hidden)
    assert output.shape == (2, 5, 3)
    output.sum().backward()
    assert hidden.grad is not None


@pytest.mark.parametrize("head_type", [BoundedQuantileHead, PositiveQuantileHead])
def test_head_rejects_non_positive_input_dimension(head_type):
    with pytest.raises(ValueError, match="positive"):
        head_type(0)
```

^- [x] **Step 2: Verify RED**

Run: `.venv/bin/pytest tests/test_quantile_heads.py -v`

Expected: import failure for `tristatelite.models.quantile_heads`.

^- [x] **Step 3: Implement heads without sorting**

Use one `nn.Linear(in_dim, 3)` per head. Split raw outputs with
`m, lower_raw, upper_raw = self.projection(hidden).unbind(dim=-1)` and implement
the exact formulas from the Phase 2B design. Stack `(q05, q50, q95)` on the final
dimension.

^- [x] **Step 4: Verify GREEN and commit**

Run:

```bash
.venv/bin/pytest tests/test_quantile_heads.py -v
.venv/bin/ruff check src/tristatelite/models tests/test_quantile_heads.py
```

Commit:

```bash
git add src/tristatelite/models tests/test_quantile_heads.py
git commit -m "feat: add ordered quantile heads"
```

### Task 4: Strictly Validated Pinball Loss

**Files:**
- Create: `src/tristatelite/losses/__init__.py`
- Create: `src/tristatelite/losses/pinball.py`
- Create: `tests/test_losses.py`

**Interfaces:**
- Produces: `pinball_loss(pred: Tensor, target: Tensor, quantiles: Tensor) -> Tensor`

^- [x] **Step 1: Write failing numeric tests**

Create `tests/test_losses.py` with:

```python
import pytest
import torch

from tristatelite.losses.pinball import pinball_loss


QUANTILES = torch.tensor([0.05, 0.50, 0.95])


def test_pinball_is_zero_for_exact_prediction():
    target = torch.tensor([2.0, 4.0])
    prediction = target[:, None].expand(-1, 3)
    assert pinball_loss(prediction, target, QUANTILES).item() == 0.0


def test_pinball_matches_manual_underprediction():
    prediction = torch.zeros(1, 3)
    target = torch.tensor([2.0])
    expected = 2.0 * (0.05 + 0.50 + 0.95) / 3
    assert pinball_loss(prediction, target, QUANTILES).item() == pytest.approx(expected)
```

Parameterize invalid cases for prediction last dimension not 3, target shape not
matching prediction leading dimensions, non-finite quantiles, unordered quantiles,
and quantiles outside `(0, 1)`.

^- [x] **Step 2: Verify RED**

Run: `.venv/bin/pytest tests/test_losses.py -v`

Expected: import failure for `tristatelite.losses.pinball`.

^- [x] **Step 3: Implement validation and loss**

Normalize a target shaped `[..., 1]` with `squeeze(-1)`, require equality with
`pred.shape[:-1]`, and move quantiles to prediction device and dtype only after
validating the supplied values. Compute:

```python
error = target.unsqueeze(-1) - pred
loss = torch.maximum(quantiles * error, (quantiles - 1.0) * error)
return loss.mean()
```

^- [x] **Step 4: Verify GREEN and commit**

Run:

```bash
.venv/bin/pytest tests/test_losses.py -v
.venv/bin/ruff check src/tristatelite/losses tests/test_losses.py
```

Commit:

```bash
git add src/tristatelite/losses tests/test_losses.py
git commit -m "feat: add validated pinball loss"
```

### Task 5: Load-Conditioned Physics Consistency

**Files:**
- Create: `src/tristatelite/losses/consistency.py`
- Modify: `tests/test_losses.py`

**Interfaces:**
- Produces: `physics_consistency_loss(soc_median, soh_median, log_tte_median, q_ref_ah, i_eff_60s, current_cv_60s, *, minimum_current_a=0.05, maximum_tte_s=604800.0) -> tuple[Tensor, dict[str, Tensor]]`

^- [x] **Step 1: Write failing matching-physics and diagnostics tests**

Add:

```python
from tristatelite.losses.consistency import physics_consistency_loss


def test_consistency_is_zero_for_matching_physical_tte():
    soc = torch.tensor([0.5])
    soh = torch.tensor([0.8])
    q_ref = torch.tensor([2.0])
    current = torch.tensor([1.0])
    physical_tte = soc * soh * q_ref / current * 3600
    loss, diagnostics = physics_consistency_loss(
        soc, soh, torch.log1p(physical_tte), q_ref, current, torch.zeros(1)
    )
    assert loss.item() == pytest.approx(0.0, abs=1e-7)
    assert diagnostics["active_fraction"].item() == 1.0
    assert diagnostics["median_physical_tte_s"].item() == pytest.approx(2880.0)
```

^- [x] **Step 2: Write failing masking, stability, and gradient tests**

Cover all of these cases:

```python
def test_low_current_is_masked_with_differentiable_zero():
    log_tte = torch.tensor([1.0], requires_grad=True)
    loss, diagnostics = physics_consistency_loss(
        torch.tensor([0.5]), torch.tensor([0.9]), log_tte,
        torch.tensor([2.0]), torch.tensor([0.01]), torch.tensor([0.0])
    )
    loss.backward()
    assert loss.item() == 0.0
    assert log_tte.grad is not None
    assert log_tte.grad.item() == 0.0
    assert diagnostics["active_fraction"].item() == 0.0


def test_variable_current_gets_lower_weight():
    common = (
        torch.tensor([0.5]), torch.tensor([0.9]), torch.tensor([1.0]),
        torch.tensor([2.0]), torch.tensor([1.0]),
    )
    _, stable = physics_consistency_loss(*common, torch.tensor([0.0]))
    _, variable = physics_consistency_loss(*common, torch.tensor([1.0]))
    assert variable["mean_weight"] < stable["mean_weight"]


def test_consistency_gradients_reach_all_medians():
    soc = torch.tensor([0.4], requires_grad=True)
    soh = torch.tensor([0.8], requires_grad=True)
    log_tte = torch.tensor([1.0], requires_grad=True)
    loss, _ = physics_consistency_loss(
        soc, soh, log_tte, torch.tensor([2.0]),
        torch.tensor([1.0]), torch.tensor([0.0]),
    )
    loss.backward()
    for value in (soc, soh, log_tte):
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()
        assert value.grad.abs().sum() > 0
```

Also test that NaN physics is inactive, shape mismatch raises `ValueError`, and
non-positive thresholds raise `ValueError`.

^- [x] **Step 3: Verify RED**

Run: `.venv/bin/pytest tests/test_losses.py -v`

Expected: import failure for `tristatelite.losses.consistency`.

^- [x] **Step 4: Implement shape-safe active masking**

Require every input tensor to have exactly the same shape. Build an active mask
from finiteness and current threshold. Replace inactive values with finite safe
constants before division, compute physical TTE and weights, then mask weighted
errors:

```python
denominator = active.sum().clamp_min(1).to(log_tte_median.dtype)
loss = torch.where(active, weight * error, torch.zeros_like(error)).sum() / denominator
```

For an all-inactive batch, include `log_tte_median.sum() * 0.0` in the returned
loss so backward remains valid. Compute diagnostics under `torch.no_grad()`; return
zero scalar diagnostics when no row is active.

^- [x] **Step 5: Verify GREEN and commit**

Run:

```bash
.venv/bin/pytest tests/test_losses.py -v
.venv/bin/ruff check src/tristatelite/losses tests/test_losses.py
```

Commit:

```bash
git add src/tristatelite/losses/consistency.py tests/test_losses.py
git commit -m "feat: add physical TTE consistency loss"
```

### Task 6: Real NASA Contract Regression and Phase Verification

**Files:**
- Modify only if verification exposes a defect in files already listed above.
- Produce ignored artifact: `data/processed/nasa-randomized-phase2b-smoke/`

**Interfaces:**
- Consumes the real archive with audited SHA256.
- Produces a leakage-audited smoke artifact and model-ready window item.

^- [x] **Step 1: Run the complete suite and lint**

Run:

```bash
.venv/bin/pytest -q
.venv/bin/ruff check src tests scripts
git diff --check
```

Expected: all tests pass, Ruff reports no errors, and no whitespace errors exist.

^- [x] **Step 2: Build a fresh real NASA smoke artifact**

Run:

```bash
.venv/bin/python scripts/build_dataset.py \
  --config configs/data/nasa_randomized.yaml \
  --archive data/raw/battery_alt_dataset.zip \
  --output data/processed/nasa-randomized-phase2b-smoke \
  --split-seed 2026 \
  --protocol calibrated \
  --limit-batteries 6 \
  --limit-cycles 3
```

Expected: status `complete`, 6 batteries, 18 accepted cycles, and passed leakage
audit.

^- [x] **Step 3: Verify real units and construct a window**

Read the artifact and assert in a one-off verification command:

- raw current is finite, differs from `current_a__scaled`, and has a median between
  `0.05 A` and `10 A`;
- `i_eff_60s` is finite and exceeds `0.05 A` for at least 95% of samples;
- `current_cv_60s` is finite;
- every `MODEL_FEATURES` column exists and is finite;
- leakage audit status is `passed`;
- a training `BatteryWindowDataset` item has fast shape `[128, 12]`, slow shape
  `[8, 8]`, and physics keys `i_eff_60s`, `current_cv_60s`, `q_ref_ah`.

^- [x] **Step 4: Run final verification after any fix**

Run again:

```bash
.venv/bin/pytest -q
.venv/bin/ruff check src tests scripts
git status --short
```

Expected: green suite and a clean worktree after all intentional commits.

The phase ends locally after verified commits. Pushing the stacked branch and
opening a PR against `agent/phase2-nasa-adapter` are separate publication actions.
