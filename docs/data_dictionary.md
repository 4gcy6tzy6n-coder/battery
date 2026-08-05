# TriStateLite Phase 2A Data Dictionary

All sample rows describe one causally resampled second inside a maximal contiguous
NASA randomized-dataset discharge (`mode == -1`). Current is positive during
discharge. No sample uses a future observation to fill an earlier timestamp.

| Field | Unit | Definition |
|---|---:|---|
| `battery_id` | — | Physical source file stem, such as `battery00`. |
| `cycle_id` | — | Stable battery-local discharge identifier. |
| `cycle_index` | — | Zero-based accepted discharge order within a battery. |
| `timestamp_s` | s | Floored lifetime timestamp; strictly increasing per cycle. |
| `elapsed_s` | s | Seconds since the first canonical sample in the cycle. |
| `voltage_v` | V | Load voltage, last observation per second, then bounded forward-fill. |
| `current_a` | A | Positive discharge current under the same causal fill rule. |
| `temperature_c` | °C | Battery temperature; missing values may remain missing. |
| `temperature_available` | bool | Whether temperature was directly observed in that second. |
| `mission_type` | — | NASA mission code: `0` reference, `1` regular mission. |
| `phase` | — | Always `discharge` in Phase 2A. |
| `soc_target` | fraction | `1 - cumulative_Ah / cycle_delivered_Ah`, clipped to `[0, 1]`. |
| `soh_target` | fraction | Cycle delivered Ah divided by the protocol reference Ah. |
| `tte_seconds` | s | Remaining observed discharge time; zero at cycle end. |
| `log_tte_target` | — | `log1p(tte_seconds)`. |
| `q_ref_ah` | Ah | Median delivered Ah of the earliest three valid calibration cycles, or an explicit global reference. |

Cycle summaries describe completed cycles. A summary carrying cycle index `k` is
eligible only for a prediction from a cycle with index greater than `k`.
