import pandas as pd

AUDITED_COLUMNS = [
    "start_time",
    "time",
    "mode",
    "voltage_charger",
    "temperature_battery",
    "voltage_load",
    "current_load",
    "temperature_mosfet",
    "temperature_resistor",
    "mission_type",
]


def synthetic_lifetime_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["2022-01-01 00:00:00", 0.0, 0, 8.4, 25.0, None, None, None, None, None],
            ["2022-01-01 00:00:00", 1.0, -1, 8.3, "  NAN", 8.2, 2.5, 26.0, 26.0, 0],
            ["2022-01-02 00:00:00", 2.0, -1, 8.2, 25.2, 8.1, 2.5, 26.1, 26.1, 0],
            ["2022-01-02 00:00:00", 3.0, -1, 8.1, 25.3, 8.0, 2.5, 26.2, 26.2, 0],
            ["2022-01-02 00:00:00", 4.0, 0, 8.0, 25.3, None, None, None, None, None],
            ["2022-01-02 00:00:00", 5.0, -1, 8.0, 25.4, 7.9, 10.0, 27.0, 27.0, 1],
            ["2022-01-02 00:00:00", 6.0, -1, 7.9, 25.5, 7.8, 10.0, 27.1, 27.1, 1],
        ],
        columns=AUDITED_COLUMNS,
    )
