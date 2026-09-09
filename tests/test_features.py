"""Feature builders."""

from __future__ import annotations

import pandas as pd

from ev_s6e9.data import synth
from ev_s6e9.features import add_derived, encode_target, prep_x, split_xy
from ev_s6e9.schema import CAT_COLS, FEATURE_COLS, TARGET


def test_encode_target_yes_no():
    s = pd.Series(["Yes", "No", "yes", "NO"])
    assert encode_target(s).tolist() == [1, 0, 1, 0]


def test_encode_target_01():
    assert encode_target(pd.Series([0, 1, 1])).tolist() == [0, 1, 1]


def test_prep_x_cols_and_derived():
    df = synth(20, seed=3)
    x = prep_x(df)
    assert "id" not in x.columns and TARGET not in x.columns
    assert "charging_total" in x.columns
    assert (
        x["charging_total"]
        == df["Charging_Stations_Near_Home"] + df["Charging_Stations_Near_Work"]
    ).all()
    for c in CAT_COLS:
        assert str(x[c].dtype) == "category"
    for c in FEATURE_COLS:
        assert c in x.columns


def test_add_derived_missing_ok():
    out = add_derived(pd.DataFrame({"Age": [1, 2]}))
    assert "charging_total" not in out.columns


def test_split_xy_len():
    x, y = split_xy(synth(15, seed=2))
    assert len(x) == len(y) == 15
    assert set(y.unique()) <= {0, 1}
