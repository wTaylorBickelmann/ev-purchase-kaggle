"""Tiny feature builders."""

from __future__ import annotations

import pandas as pd

from ev_s6e9.schema import CAT_COLS, FEATURE_COLS, ID_COL, NUM_COLS, TARGET, check_cols


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
