"""Feature builders: LightGBM baseline + Deotte FeatureBuilder."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from ev_s6e9.schema import CAT_COLS, FEATURE_COLS, ID_COL, NUM_COLS, TARGET, check_cols

HELPER_COLS = ["worry_score", "chargers_total", "income_x_subsidy", "concern_x_subsidy"]
RECIPE_COL = "recipe_score"
FREQ_COLS = ["Annual_Income_USD", "Daily_Commute_km"]
FREQ_SUFFIX = "_cnt"


def encode_target(s: pd.Series) -> pd.Series:
    """Map Yes/No / 1/0 / True/False to int {0,1}."""
    if pd.api.types.is_numeric_dtype(s):
        return (pd.to_numeric(s, errors="coerce") > 0).astype(int)
    m = s.astype(str).str.strip().str.lower()
    return m.isin(["1", "yes", "true", "y"]).astype(int)


def add_derived(x: pd.DataFrame) -> pd.DataFrame:
    """One cheap interaction: public chargers at home + work."""
    x = x.copy()
    home, work = "Charging_Stations_Near_Home", "Charging_Stations_Near_Work"
    if home in x.columns and work in x.columns:
        x["charging_total"] = pd.to_numeric(x[home], errors="coerce") + pd.to_numeric(
            x[work], errors="coerce"
        )
    return x


def prep_x(df: pd.DataFrame) -> pd.DataFrame:
    """Drop id/target, coerce types, add derived. LightGBM reads category dtype."""
    check_cols(df.columns, FEATURE_COLS, "features")
    x = df.drop(columns=[c for c in (ID_COL, TARGET) if c in df.columns], errors="ignore")
    for c in NUM_COLS:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    for c in CAT_COLS:
        x[c] = x[c].astype("string").astype("category")
    return add_derived(x)


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return prep_x(df), encode_target(df[TARGET])


def _yes(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().eq("yes").astype(float)


class FeatureBuilder:
    """Deotte Fable 5.1: helper features, recipe score/logit, category codes."""

    def __init__(self, freq: bool = False) -> None:
        self.freq = freq
        self._cat_dtypes: dict[str, pd.CategoricalDtype] = {}
        self._freq_maps: dict[str, pd.Series] = {}
        self._fitted = False

    def fit(self, train: pd.DataFrame, test: pd.DataFrame | None = None) -> FeatureBuilder:
        frames = [train]
        if test is not None:
            frames.append(test)
        combined = pd.concat(frames, ignore_index=True)
        for c in CAT_COLS:
            cats = combined[c].astype("string").dropna().unique()
            self._cat_dtypes[c] = pd.CategoricalDtype(categories=sorted(cats))
        if self.freq:
            for c in FREQ_COLS:
                self._freq_maps[c] = pd.to_numeric(combined[c], errors="coerce").value_counts()
        self._fitted = True
        return self

    def _require_fit(self) -> None:
        if not self._fitted:
            raise RuntimeError("FeatureBuilder.fit() required before transform")

    @staticmethod
    def recipe_score(df: pd.DataFrame) -> pd.Series:
        income = pd.to_numeric(df["Annual_Income_USD"], errors="coerce") / 1e5
        concern = pd.to_numeric(df["Environmental_Concern_Level"], errors="coerce")
        subsidy = _yes(df["Subsidy_Available"])
        anxiety = df["Range_Anxiety_Level"].astype(str)
        return (
            1.2 * income
            + 0.6 * concern
            + 2.0 * subsidy
            - 1.0 * anxiety.eq("Medium").astype(float)
            - 3.0 * anxiety.eq("High").astype(float)
        )

    @staticmethod
    def recipe_logit(df: pd.DataFrame) -> np.ndarray:
        score = FeatureBuilder.recipe_score(df).to_numpy(dtype=float)
        p = np.clip(norm.cdf(score - 5.5), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    def _helper_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        commute = pd.to_numeric(df["Daily_Commute_km"], errors="coerce")
        home = pd.to_numeric(df["Charging_Stations_Near_Home"], errors="coerce")
        work = pd.to_numeric(df["Charging_Stations_Near_Work"], errors="coerce")
        income = pd.to_numeric(df["Annual_Income_USD"], errors="coerce") / 1e5
        concern = pd.to_numeric(df["Environmental_Concern_Level"], errors="coerce")
        subsidy = _yes(df["Subsidy_Available"])
        home_chg = _yes(df["Home_Charging_Possible"])
        return pd.DataFrame(
            {
                "worry_score": commute - 5 * home - 5 * work - 150 * home_chg,
                "chargers_total": home + work,
                "income_x_subsidy": income * subsidy,
                "concern_x_subsidy": concern * subsidy,
            },
            index=df.index,
        )

    def transform(self, df: pd.DataFrame, *, with_recipe: bool = False) -> pd.DataFrame:
        self._require_fit()
        check_cols(df.columns, FEATURE_COLS, "features")
        helpers = self._helper_frame(df)
        x = df[FEATURE_COLS].copy()
        for c in NUM_COLS:
            x[c] = pd.to_numeric(x[c], errors="coerce")
        for c in CAT_COLS:
            x[c] = df[c].astype("string").astype(self._cat_dtypes[c]).cat.codes.astype(np.int16)
        for c in HELPER_COLS:
            x[c] = helpers[c]
        if self.freq:
            for c in FREQ_COLS:
                vals = pd.to_numeric(df[c], errors="coerce")
                x[c + FREQ_SUFFIX] = vals.map(self._freq_maps[c]).fillna(0).astype(np.float32)
        if with_recipe:
            x[RECIPE_COL] = self.recipe_score(df)
        return x
