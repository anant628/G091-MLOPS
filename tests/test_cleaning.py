import pandas as pd

from src.opd.cleaning import clean_train

PARAMS = {"cleaning": {
    "sentinel_column": "lab_tests_pending",
    "sentinel_value": -1,
    "typo_threshold": 400,
    "typo_factor": 10,
    "stuck_column": "doctors_on_duty",
    "stuck_start": "2026-03-29",
    "stuck_end": "2026-04-19",
}}


def make_df():
    return pd.DataFrame({
        "record_id": ["R1", "R2", "R2", "R3", "R4"],
        "date": ["2026-01-01", "2026-01-02", "2026-01-02", "2026-03-30", "2026-05-01"],
        "doctors_on_duty": [4, 5, 5, 8, 8],
        "lab_tests_pending": [10, -1, -1, 5, 7],
        "wait_min": [100, 90, 90, 120, 1500],
    })


def test_duplicates_removed():
    assert len(clean_train(make_df(), PARAMS)) == 4


def test_sentinel_becomes_missing():
    out = clean_train(make_df(), PARAMS)
    assert out["lab_tests_pending"].isna().sum() == 1
    assert (out["lab_tests_pending"].dropna() >= 0).all()


def test_stuck_sensor_dates_blanked():
    out = clean_train(make_df(), PARAMS)
    assert out["doctors_on_duty"].isna().sum() == 1   # only the row on 2026-03-30


def test_target_typo_fixed():
    out = clean_train(make_df(), PARAMS)
    assert out["wait_min"].max() == 150