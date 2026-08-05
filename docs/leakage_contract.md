# Phase 2A Leakage Contract

The dataset build is valid only when all of these executable rules pass:

1. A physical `battery_id` belongs to exactly one of train, validation, or test.
2. Every sample inherits its split exclusively from `battery_id`; sample and cycle
   attributes never influence assignment.
3. Imputation medians, IQR scales, and any other fitted preprocessing statistics
   use training batteries only.
4. A slow-history row for cycle `k` is visible only while predicting cycles with
   index greater than `k` from the same battery.
5. Current-cycle delivered capacity, targets, final values, and future-derived
   fields are forbidden as model input features.
6. `(battery_id, cycle_id, timestamp_s)` sample keys are globally unique and may
   not occur in multiple splits.
7. All rolling features and missing-value fills are backward-looking. Evaluation
   splits reuse frozen training preprocessing artifacts.

`audit_leakage` fails the build before artifact publication if any rule it can
verify from the manifest, samples, history, and scaler provenance is violated.
