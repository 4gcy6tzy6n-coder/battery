import numpy as np
import pandas as pd

from tristatelite.data.scaling import fit_scaler, transform_features


def test_scaler_uses_only_supplied_training_values_and_preserves_missingness_flag():
    train = pd.DataFrame(
        {
            "battery_id": ["battery00"] * 5,
            "voltage_v": [-2.0, -1.0, 0.0, 1.0, 2.0],
            "temperature_c": [20.0, np.nan, 22.0, 24.0, 26.0],
            "temperature_available": [True, False, True, True, True],
        }
    )
    held_out = pd.DataFrame(
        {
            "battery_id": ["battery20", "battery20"],
            "voltage_v": [100.0, 101.0],
            "temperature_c": [np.nan, 120.0],
            "temperature_available": [False, True],
        }
    )

    artifact = fit_scaler(train, ["voltage_v", "temperature_c"])
    transformed = transform_features(held_out, artifact)

    assert artifact["features"]["voltage_v"] == {"median": 0.0, "iqr": 2.0}
    assert artifact["features"]["temperature_c"]["median"] == 23.0
    assert artifact["fitted_battery_ids"] == ["battery00"]
    assert transformed["voltage_v"].min() == 10.0
    assert transformed["temperature_c"].iloc[0] == 0.0
    assert transformed["temperature_available"].tolist() == [False, True]
