"""Deotte FeatureBuilder: recipe score/logit and helper features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ev_s6e9.data import synth
from ev_s6e9.features import HELPER_COLS, FeatureBuilder, RECIPE_COL


def _row(**kw) -> pd.DataFrame:
    base = synth(1, seed=0, target=True).iloc[0].to_dict()
    base.update(kw)
    return pd.DataFrame([base])


def test_recipe_score_formula():
    df = _row(
        Annual_Income_USD=100_000,
        Environmental_Concern_Level=5,
        Subsidy_Available="Yes",
        Range_Anxiety_Level="Low",
    )
    score = FeatureBuilder.recipe_score(df).iloc[0]
    assert score == pytest.approx(1.2 * 1.0 + 0.6 * 5 + 2.0)


def test_recipe_score_range_anxiety_penalties():
    low = FeatureBuilder.recipe_score(_row(Range_Anxiety_Level="Low")).iloc[0]
    med = FeatureBuilder.recipe_score(_row(Range_Anxiety_Level="Medium")).iloc[0]
    high = FeatureBuilder.recipe_score(_row(Range_Anxiety_Level="High")).iloc[0]
    assert med == low - 1.0
    assert high == low - 3.0


def test_recipe_logit_monotone_and_bounded():
    df = _row(Annual_Income_USD=50_000, Environmental_Concern_Level=3, Subsidy_Available="No")
    logit = FeatureBuilder.recipe_logit(df)[0]
    assert np.isfinite(logit)
    high = FeatureBuilder.recipe_logit(
        _row(Annual_Income_USD=200_000, Environmental_Concern_Level=10, Subsidy_Available="Yes")
    )[0]
    assert high > logit


def test_helper_features():
    df = _row(
        Daily_Commute_km=20.0,
        Charging_Stations_Near_Home=2,
        Charging_Stations_Near_Work=3,
        Home_Charging_Possible="Yes",
        Annual_Income_USD=80_000,
        Environmental_Concern_Level=7,
        Subsidy_Available="Yes",
    )
    fb = FeatureBuilder().fit(df)
    x = fb.transform(df)
    assert list(HELPER_COLS) == [
        "worry_score",
        "chargers_total",
        "income_x_subsidy",
        "concern_x_subsidy",
    ]
    assert x["worry_score"].iloc[0] == pytest.approx(20 - 5 * 2 - 5 * 3 - 150)
    assert x["chargers_total"].iloc[0] == 5
    assert x["income_x_subsidy"].iloc[0] == pytest.approx(0.8)
    assert x["concern_x_subsidy"].iloc[0] == pytest.approx(7.0)


def test_category_codes_shared_train_test():
    tr = synth(10, seed=0, target=True)
    te = synth(5, seed=1, target=False, start_id=100)
    fb = FeatureBuilder().fit(tr, te)
    x_tr = fb.transform(tr)
    x_te = fb.transform(te)
    for c in ("Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible", "Subsidy_Available", "Range_Anxiety_Level"):
        assert x_tr[c].dtype == np.int16 or str(x_tr[c].dtype) == "int16"
        assert x_te[c].min() >= -1


def test_transform_with_recipe():
    df = synth(8, seed=2, target=True)
    fb = FeatureBuilder().fit(df)
    x = fb.transform(df, with_recipe=True)
    assert RECIPE_COL in x.columns
    assert (x[RECIPE_COL] == FeatureBuilder.recipe_score(df)).all()
