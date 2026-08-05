import pandas as pd
import pytest

from tristatelite.data.audit import audit_leakage
from tristatelite.data.split import make_group_split

BATTERIES = [f"battery{index:02d}" for index in range(26)]


def test_group_split_is_repeatable_disjoint_and_complete():
    first = make_group_split(BATTERIES, seed=2026)
    second = make_group_split(list(reversed(BATTERIES)), seed=2026)

    assert first == second
    train, val, test = map(
        set, (first.train_batteries, first.val_batteries, first.test_batteries)
    )
    assert (len(train), len(val), len(test)) == (18, 4, 4)
    assert not train & val and not train & test and not val & test
    assert train | val | test == set(BATTERIES)


def _audit_frames(manifest):
    rows = []
    for split, ids in (
        ("train", manifest.train_batteries),
        ("val", manifest.val_batteries),
        ("test", manifest.test_batteries),
    ):
        for battery_id in ids:
            rows.append(
                {
                    "battery_id": battery_id,
                    "cycle_id": f"{battery_id}-discharge-00001",
                    "timestamp_s": 1.0,
                    "split": split,
                }
            )
    samples = pd.DataFrame(rows)
    history = pd.DataFrame(
        {
            "battery_id": [manifest.train_batteries[0]],
            "current_cycle_index": [2],
            "history_cycle_index": [1],
            "mean_voltage_v": [7.0],
        }
    )
    return samples, history


def test_leakage_audit_accepts_battery_inheritance_and_past_only_history():
    manifest = make_group_split(BATTERIES)
    samples, history = _audit_frames(manifest)

    report = audit_leakage(manifest, samples, history, set(manifest.train_batteries))

    assert report["status"] == "passed"
    assert report["battery_overlap_count"] == 0


@pytest.mark.parametrize(
    "mutation",
    ["unassigned", "wrong_split", "heldout_scaler", "future_history", "forbidden", "duplicate"],
)
def test_leakage_audit_rejects_each_contract_violation(mutation):
    manifest = make_group_split(BATTERIES)
    samples, history = _audit_frames(manifest)
    scaler_ids = set(manifest.train_batteries)
    if mutation == "unassigned":
        samples.loc[0, "battery_id"] = "battery99"
    elif mutation == "wrong_split":
        samples.loc[0, "split"] = "test"
    elif mutation == "heldout_scaler":
        scaler_ids.add(manifest.test_batteries[0])
    elif mutation == "future_history":
        history.loc[0, "history_cycle_index"] = 2
    elif mutation == "forbidden":
        history["future_target"] = 1.0
    elif mutation == "duplicate":
        samples = pd.concat([samples, samples.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError):
        audit_leakage(manifest, samples, history, scaler_ids)
