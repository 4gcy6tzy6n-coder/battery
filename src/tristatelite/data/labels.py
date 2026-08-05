"""Physics-derived battery targets and completed-cycle summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd


def integrate_discharge_ah(time_s: np.ndarray, current_a: np.ndarray) -> np.ndarray:
    """Integrate positive discharge current with the cumulative trapezoid rule."""
    time = np.asarray(time_s, dtype=float)
    current = np.asarray(current_a, dtype=float)
    if time.ndim != 1 or current.ndim != 1 or len(time) != len(current):
        raise ValueError("time and current must be equal-length one-dimensional arrays")
    if len(time) == 0 or not np.isfinite(time).all() or not np.isfinite(current).all():
        raise ValueError("time and current must be finite and non-empty")
    if np.any(current < 0):
        raise ValueError("negative current is invalid for discharge-positive data")
    delta_s = np.diff(time)
    if np.any(delta_s <= 0):
        raise ValueError("time must be strictly increasing")
    increments = delta_s * (current[:-1] + current[1:]) * 0.5 / 3600.0
    return np.r_[0.0, np.cumsum(increments)]


def _reference_capacity(
    cycles: list[pd.DataFrame], capacities: list[float], protocol: str, global_q_ref_ah: float | None
) -> float:
    if global_q_ref_ah is not None:
        q_ref = float(global_q_ref_ah)
    elif protocol == "calibrated":
        if len(cycles) < 3:
            raise ValueError("calibrated protocol requires at least three valid cycles")
        order = np.argsort([int(cycle["cycle_index"].iloc[0]) for cycle in cycles])
        q_ref = float(np.median([capacities[index] for index in order[:3]]))
    else:
        raise ValueError("a positive global_q_ref_ah is required for the global protocol")
    if not np.isfinite(q_ref) or q_ref <= 0:
        raise ValueError("q_ref_ah must be positive and finite")
    return q_ref


def _mean_voltage_slope(cycle: pd.DataFrame) -> float:
    time = cycle["elapsed_s"].to_numpy(dtype=float)
    voltage = cycle["voltage_v"].to_numpy(dtype=float)
    return float(np.mean(np.diff(voltage) / np.diff(time)))


def label_battery_cycles(
    cycles: list[pd.DataFrame],
    protocol: str,
    global_q_ref_ah: float | None = None,
) -> tuple[list[pd.DataFrame], pd.DataFrame]:
    """Attach sample targets and return summaries usable only by later cycles."""
    if not cycles:
        return [], pd.DataFrame()

    cumulative = [
        integrate_discharge_ah(
            cycle["elapsed_s"].to_numpy(dtype=float),
            cycle["current_a"].to_numpy(dtype=float),
        )
        for cycle in cycles
    ]
    capacities = [float(values[-1]) for values in cumulative]
    if any(capacity <= 0 for capacity in capacities):
        raise ValueError("all cycle capacities must be positive")
    q_ref = _reference_capacity(cycles, capacities, protocol, global_q_ref_ah)

    labeled: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    all_unclipped: list[np.ndarray] = []
    for cycle, discharged_ah, delivered_ah in zip(cycles, cumulative, capacities, strict=True):
        result = cycle.copy()
        elapsed = result["elapsed_s"].to_numpy(dtype=float)
        soc = 1.0 - discharged_ah / delivered_ah
        soh = np.full(len(result), delivered_ah / q_ref)
        tte = elapsed[-1] - elapsed
        all_unclipped.extend([soc, soh])
        result["soc_target"] = np.clip(soc, 0.0, 1.0)
        result["soh_target"] = np.clip(soh, 0.0, 1.0)
        result["tte_seconds"] = tte
        result["log_tte_target"] = np.log1p(tte)
        result["q_ref_ah"] = q_ref
        labeled.append(result)

        temperature = result["temperature_c"].dropna()
        cycle_index = int(result["cycle_index"].iloc[0])
        summary_rows.append(
            {
                "battery_id": str(result["battery_id"].iloc[0]),
                "cycle_id": str(result["cycle_id"].iloc[0]),
                "cycle_index": cycle_index,
                "available_after_cycle_index": cycle_index,
                "protocol": protocol,
                "delivered_ah": delivered_ah,
                "duration_s": float(elapsed[-1] - elapsed[0]),
                "mean_voltage_v": float(result["voltage_v"].mean()),
                "mean_current_a": float(result["current_a"].mean()),
                "mean_temperature_c": float(temperature.mean()) if len(temperature) else np.nan,
                "current_std_a": float(result["current_a"].std(ddof=0)),
                "temperature_rise_c": (
                    float(temperature.iloc[-1] - temperature.iloc[0])
                    if len(temperature)
                    else np.nan
                ),
                "voltage_slope_v_per_s": _mean_voltage_slope(result),
                "q_ref_ah": q_ref,
            }
        )

    values = np.concatenate(all_unclipped)
    outside_fraction = float(np.mean((values < -0.02) | (values > 1.05 + 1e-12)))
    if outside_fraction > 0.01:
        raise ValueError(
            f"{outside_fraction:.2%} of SOC/SOH labels lie outside audited range [-0.02, 1.05]"
        )
    return labeled, pd.DataFrame(summary_rows)
