"""Load competition CSVs, download via Kaggle CLI, tiny synth for tests."""

from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from ev_s6e9.paths import DATA_RAW, SAMPLE_CSV, TEST_CSV, TRAIN_CSV
from ev_s6e9.schema import COMPETITION, ID_COL, TARGET, TEST_COLS, TRAIN_COLS, check_cols

_GENDERS = ["Male", "Female"]
_CITIES = ["Urban", "Suburban", "Rural"]
_CARS = ["Sedan", "SUV", "Hatchback", "Truck"]
_YN = ["Yes", "No"]
_ANX = ["Low", "Medium", "High"]


def load_train(path: Path | None = None) -> pd.DataFrame:
    p = path or TRAIN_CSV
    df = pd.read_csv(p)
    check_cols(df.columns, TRAIN_COLS, "train")
    return df


def load_test(path: Path | None = None) -> pd.DataFrame:
    p = path or TEST_CSV
    df = pd.read_csv(p)
    check_cols(df.columns, TEST_COLS, "test")
    return df


def load_sample(path: Path | None = None) -> pd.DataFrame:
    p = path or SAMPLE_CSV
    return pd.read_csv(p)


def download(dest: Path | None = None) -> Path:
    """`kaggle competitions download -c playground-series-s6e9` into data/raw."""
    dest = dest or DATA_RAW
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [
        "kaggle",
        "competitions",
        "download",
        "-c",
        COMPETITION,
        "-p",
        str(dest),
    ]
    subprocess.run(cmd, check=True)
    for z in dest.glob("*.zip"):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(dest)
        z.unlink()
    return dest


def synth(n: int = 200, seed: int = 0, *, target: bool = True, start_id: int = 0) -> pd.DataFrame:
    """Schema-accurate smoke data (not the real competition files)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            ID_COL: np.arange(start_id, start_id + n),
            "Age": rng.integers(25, 71, n),
            "Annual_Income_USD": rng.integers(25_000, 180_000, n),
            "Daily_Commute_km": rng.uniform(1, 80, n).round(2),
            "Number_of_Cars_Owned": rng.integers(0, 4, n),
            "Charging_Stations_Near_Home": rng.integers(0, 12, n),
            "Charging_Stations_Near_Work": rng.integers(0, 12, n),
            "Environmental_Concern_Level": rng.integers(1, 11, n),
            "Gender": rng.choice(_GENDERS, n),
            "City_Type": rng.choice(_CITIES, n),
            "Current_Car_Type": rng.choice(_CARS, n),
            "Home_Charging_Possible": rng.choice(_YN, n),
            "Subsidy_Available": rng.choice(_YN, n),
            "Range_Anxiety_Level": rng.choice(_ANX, n),
        }
    )
    if target:
        p = 0.25 + 0.04 * df["Environmental_Concern_Level"] - 0.08 * (
            df["Range_Anxiety_Level"] == "High"
        )
        df[TARGET] = (rng.random(n) < p.clip(0.05, 0.95)).astype(int)
        check_cols(df.columns, TRAIN_COLS, "synth")
    else:
        check_cols(df.columns, TEST_COLS, "synth")
    return df


def write_synth_raw(dest: Path | None = None, n_train: int = 400, n_test: int = 100) -> Path:
    dest = dest or DATA_RAW
    dest.mkdir(parents=True, exist_ok=True)
    tr = synth(n_train, seed=0, target=True)
    te = synth(n_test, seed=1, target=False, start_id=n_train)
    tr.to_csv(dest / "train.csv", index=False)
    te.to_csv(dest / "test.csv", index=False)
    pd.DataFrame({ID_COL: te[ID_COL], TARGET: 0.5}).to_csv(dest / "sample_submission.csv", index=False)
    return dest


def has_real_data(raw: Path | None = None) -> bool:
    raw = raw or DATA_RAW
    return (raw / "train.csv").exists() and (raw / "test.csv").exists()
