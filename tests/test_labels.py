import numpy as np
import pandas as pd
import pytest

from tristatelite.data.labels import integrate_discharge_ah, label_battery_cycles


def _cycle(index: int, current_a: float) -> pd.DataFrame:
    elapsed = np.arange(3601, dtype=float)
    return pd.DataFrame(
        {
            "battery_id": "battery00",
            "cycle_id": f"battery00-discharge-{index:05d}",
            "cycle_index": index,
            "timestamp_s": elapsed + index * 4000,
            "elapsed_s": elapsed,
            "voltage_v": np.linspace(8.2, 5.8, len(elapsed)),
            "current_a": current_a,
            "temperature_c": np.linspace(25.0, 27.0, len(elapsed)),
            "temperature_available": True,
            "mission_type": 0.0,
            "phase": "discharge",
        }
    )


def test_integrates_one_amp_hour_cumulatively():
    time_s = np.arange(3601, dtype=float)

    cumulative = integrate_discharge_ah(time_s, np.ones_like(time_s))

    assert cumulative[0] == 0.0
    assert cumulative[-1] == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize(
    ("time_s", "current_a", "message"),
    [
        (np.array([0.0, 1.0]), np.array([1.0, -1.0]), "negative current"),
        (np.array([0.0, 2.0, 1.0]), np.ones(3), "strictly increasing"),
    ],
)
def test_integration_rejects_invalid_physics(time_s, current_a, message):
    with pytest.raises(ValueError, match=message):
        integrate_discharge_ah(time_s, current_a)


def test_labels_are_consistent_and_q_ref_uses_earliest_three_cycles():
    cycles = [_cycle(0, 1.0), _cycle(1, 0.95), _cycle(2, 1.05)]

    labeled, summaries = label_battery_cycles(cycles, protocol="calibrated")

    assert summaries["q_ref_ah"].nunique() == 1
    assert summaries["q_ref_ah"].iloc[0] == pytest.approx(1.0)
    for cycle in labeled:
        assert cycle["soc_target"].iloc[0] == pytest.approx(1.0)
        assert cycle["soc_target"].iloc[-1] == pytest.approx(0.0)
        assert cycle["tte_seconds"].iloc[-1] == 0.0
        assert (cycle["tte_seconds"].diff().dropna() <= 0).all()
        np.testing.assert_allclose(
            cycle["log_tte_target"], np.log1p(cycle["tte_seconds"])
        )
        assert cycle["soh_target"].nunique() == 1


def test_summaries_are_only_available_to_later_cycles_and_have_safe_names():
    cycles = [_cycle(0, 1.0), _cycle(1, 0.98), _cycle(2, 0.96)]

    _, summaries = label_battery_cycles(cycles, protocol="calibrated")

    assert (summaries["available_after_cycle_index"] == summaries["cycle_index"]).all()
    forbidden = ("current_cycle_capacity", "target", "future", "final_capacity")
    assert not any(token in column for column in summaries for token in forbidden)
