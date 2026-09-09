"""Lock competition headers. Do not invent column names."""

from __future__ import annotations

import pytest

from ev_s6e9.__main__ import _parser
from ev_s6e9.schema import (
    CAT_COLS,
    FEATURE_COLS,
    ID_COL,
    NUM_COLS,
    SUB_COLS,
    TARGET,
    TEST_COLS,
    TRAIN_COLS,
    check_exact,
)


def test_train_header():
    assert TRAIN_COLS == [
        "id",
        "Age",
        "Annual_Income_USD",
        "Daily_Commute_km",
        "Number_of_Cars_Owned",
        "Charging_Stations_Near_Home",
        "Charging_Stations_Near_Work",
        "Environmental_Concern_Level",
        "Gender",
        "City_Type",
        "Current_Car_Type",
        "Home_Charging_Possible",
        "Subsidy_Available",
        "Range_Anxiety_Level",
        "Will_Buy_EV",
    ]


def test_test_and_sub_headers():
    assert TEST_COLS == [c for c in TRAIN_COLS if c != TARGET]
    assert SUB_COLS == [ID_COL, TARGET]
    assert TARGET not in TEST_COLS
    assert FEATURE_COLS == NUM_COLS + CAT_COLS
    assert FEATURE_COLS == [c for c in TRAIN_COLS if c not in (ID_COL, TARGET)]


def test_check_exact_rejects_invented_names():
    with pytest.raises(ValueError, match="unexpected|must be"):
        check_exact(["id", "Age", "Fake_Col"], ["id", "Age"], "train")


def test_cli_has_download_train_predict_submit():
    p = _parser()
    for cmd in ("download", "train", "predict", "submit"):
        p.parse_args([cmd] if cmd != "submit" else ["submit", "-m", "ok"])
