# TriStateLite Phase 2B Probabilistic Core Design

## 1. Scope

Phase 2B builds the probability and physics primitives required by every later
baseline and TriStateLite model. It also corrects the processed-data contract so
model features can be scaled without destroying the physical units required by
the consistency loss.

Included:

- preservation of unit-bearing voltage, current, temperature, effective current,
  current variability, reference capacity, and time fields;
- separately named robust-scaled model inputs;
- ordered bounded quantile heads for SOC and SOH;
- ordered positive quantile heads for log-TTE;
- pinball loss;
- load-conditioned physics-consistency loss and diagnostics.

Excluded:

- GRU baselines;
- causal TCN, history GRU, and gated fusion;
- optimizer, checkpoint, training, and evaluation loops;
- zero-shot reference-capacity protocol.

## 2. Design Decision

Three approaches were considered.

1. Add only heads and losses. This is small but incorrect because the current
   processed artifact overwrites `current_a` with a robust-scaled value and derives
   `current_cv_60s` after centering current. Those values cannot support a physical
   ampere-hour equation.
2. Preserve physical channels, emit separately named scaled inputs, then implement
   the probability primitives. This is selected because it restores explicit units
   and keeps learned features separate from physical supervision.
3. Implement the complete model and training engine in the same phase. This is
   rejected because it makes probability ordering, loss numerics, physical units,
   and optimization failures difficult to isolate.

## 3. Data-to-Model Contract

Feature engineering runs on the causal, unit-bearing sample table before scaling.
It produces:

- `voltage_delta_1s` in V;
- `current_delta_1s` in A;
- `temperature_delta_1s` in °C;
- `current_mean_10s` and `current_std_10s` in A;
- `i_eff_60s` in A;
- `current_cv_60s = std_60s / (abs(mean_60s) + 1e-3)` as a dimensionless value;
- `voltage_slope_30s` in V/s;
- `temperature_slope_60s` in °C/s.

The raw columns remain unchanged in the processed artifact. The scaler fits only
training batteries and writes continuous model inputs to columns ending in
`__scaled`. `temperature_available` remains an unscaled Boolean input. The model
feature list is therefore explicit and cannot accidentally contain a unit-bearing
physics field.

The default continuous feature list is:

```text
voltage_v
current_a
temperature_c
voltage_delta_1s
current_delta_1s
temperature_delta_1s
current_mean_10s
current_std_10s
i_eff_60s
voltage_slope_30s
temperature_slope_60s
```

`transform_features(..., output_suffix="__scaled")` creates the scaled copies and
does not overwrite their sources. Existing no-suffix behavior remains available
for backwards-compatible unit tests, but dataset builds must use the suffix.

Each `BatteryWindowDataset` item exposes physics as a named mapping:

```python
{
    "i_eff_60s": Tensor,       # A
    "current_cv_60s": Tensor,  # dimensionless
    "q_ref_ah": Tensor,        # Ah
}
```

This replaces the positional physics tensor containing scaled voltage and current.

## 4. Ordered Quantile Heads

All heads produce columns in the fixed order `(0.05, 0.50, 0.95)` without sorting
after prediction.

For SOC and SOH, `BoundedQuantileHead` projects a hidden vector to `(m, l, u)` and
uses:

```text
q50 = sigmoid(m)
q05 = q50 * (1 - sigmoid(l))
q95 = q50 + (1 - q50) * sigmoid(u)
```

This guarantees `0 <= q05 <= q50 <= q95 <= 1` while preserving separate lower,
median, and upper semantics.

For the non-negative log-TTE target, `PositiveQuantileHead` uses:

```text
q50 = softplus(m)
q05 = q50 * sigmoid(l)
q95 = q50 + softplus(u)
```

This guarantees `0 <= q05 <= q50 <= q95`.

Both heads accept tensors shaped `[..., in_dim]` and return `[..., 3]`. They reject
non-positive input dimensions at construction.

## 5. Pinball Loss

`pinball_loss(pred, target, quantiles)` accepts predictions shaped `[..., 3]`,
targets shaped `[...]` or `[..., 1]`, and exactly three strictly increasing
quantiles inside `(0, 1)`. It computes:

```text
error = target - prediction
loss = max(quantile * error, (quantile - 1) * error)
```

and returns the mean over samples and quantiles. Shape errors, non-finite quantiles,
and invalid ordering raise `ValueError` instead of broadcasting silently.

## 6. Physics-Consistency Loss

The interface consumes median predictions and unscaled physics channels:

```python
physics_consistency_loss(
    soc_median: Tensor,
    soh_median: Tensor,
    log_tte_median: Tensor,
    q_ref_ah: Tensor,
    i_eff_60s: Tensor,
    current_cv_60s: Tensor,
    *,
    minimum_current_a: float = 0.05,
    maximum_tte_s: float = 604800.0,
) -> tuple[Tensor, dict[str, Tensor]]
```

For rows where all inputs are finite and `i_eff_60s > 0.05 A`:

```text
tte_phys_s = soc_median * soh_median * q_ref_ah / i_eff_60s * 3600
tte_phys_s = clamp(tte_phys_s, 0, 604800)
weight = exp(-2 * clamp(current_cv_60s, min=0))
error = abs(log_tte_median - log1p(tte_phys_s))
```

The scalar loss is the weighted-error sum divided by the number of active rows.
When there are no active rows, it returns a differentiable zero. The diagnostics
mapping contains scalar tensors named `active_fraction`, `mean_weight`, and
`median_physical_tte_s`; diagnostics are detached from autograd.

Clamping applies only to the derived physical TTE, not to model predictions.
Gradients must reach SOC, SOH, and log-TTE medians for active, non-saturated rows.

## 7. Failure Handling and Audits

- The build fails if any requested scaler source feature is absent.
- The build report records raw physical ranges and scaled feature ranges separately.
- The leakage audit rejects a fast feature list containing raw target or reference
  capacity columns.
- Dataset construction fails if required physics columns are absent.
- Loss functions reject incompatible shapes instead of relying on implicit
  broadcasting.
- NaN/Inf physics rows are inactive and included in `active_fraction` diagnostics.

## 8. Verification

Tests cover:

- future changes cannot alter engineered physical features in the past;
- scaler suffix mode preserves raw values and creates finite scaled copies;
- a synthetic dataset item exposes unscaled named physics values;
- a real NASA smoke rebuild has plausible non-negative `i_eff_60s` and finite
  current CV values;
- 1,000 random head outputs obey quantile ordering and bounds;
- exact and hand-computed pinball examples;
- zero consistency error for a matching physical prediction;
- low-current masking and differentiable all-masked zero;
- lower stability weight for higher current CV;
- gradients reach all three predicted medians;
- the complete pre-existing test suite and Ruff checks remain green.

## 9. Completion Boundary

Phase 2B is complete when the corrected real NASA smoke artifact passes its leakage
audit, the probability/loss tests pass, the complete suite passes, and the work is
committed on a new stacked feature branch. Baseline model implementation begins in
the following phase.
