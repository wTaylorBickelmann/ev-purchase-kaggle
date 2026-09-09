"""Competition column names. Do not invent extras.

train.csv (~668665 rows) and test.csv (~286571, no target) use exactly these names.
"""

from __future__ import annotations

ID_COL = "id"
TARGET = "Will_Buy_EV"

# train.csv header, in file order
TRAIN_COLS = [
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

TEST_COLS = [c for c in TRAIN_COLS if c != TARGET]
SUB_COLS = [ID_COL, TARGET]

NUM_COLS = [
    "Age",
    "Annual_Income_USD",
    "Daily_Commute_km",
    "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home",
    "Charging_Stations_Near_Work",
    "Environmental_Concern_Level",
]

CAT_COLS = [
    "Gender",
    "City_Type",
    "Current_Car_Type",
    "Home_Charging_Possible",
    "Subsidy_Available",
    "Range_Anxiety_Level",
]

FEATURE_COLS = NUM_COLS + CAT_COLS

COMPETITION = "playground-series-s6e9"


def missing_cols(cols: list[str], have) -> list[str]:
    """Return expected names that are absent from `have`."""
    return [c for c in cols if c not in have]


def check_cols(have, cols: list[str], name: str) -> None:
    miss = missing_cols(cols, have)
    if miss:
        raise ValueError(f"{name} missing columns: {miss}")


def check_exact(have, cols: list[str], name: str) -> None:
    """Require the competition header, in order — no invented names."""
    got = list(have)
    if got != list(cols):
        raise ValueError(f"{name} columns must be {list(cols)}, got {got}")


def check_submission(df) -> None:
    """Raise if a submission frame is not `id,Will_Buy_EV`."""
    if list(df.columns) != SUB_COLS:
        raise ValueError(f"submission columns must be {SUB_COLS}, got {list(df.columns)}")
    if df[ID_COL].duplicated().any():
        raise ValueError("submission has duplicate ids")
    if df[TARGET].isna().any():
        raise ValueError("submission has null probabilities")
