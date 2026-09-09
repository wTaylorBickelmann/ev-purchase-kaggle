"""Kaggle submit CLI wrapper."""

from __future__ import annotations

import subprocess
from pathlib import Path

from ev_s6e9.paths import SUB_CSV
from ev_s6e9.schema import COMPETITION, check_submission
import pandas as pd


def submit(path: Path | None = None, message: str = "lgbm baseline") -> None:
    path = path or SUB_CSV
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run predict first")
    check_submission(pd.read_csv(path))
    cmd = [
        "kaggle",
        "competitions",
        "submit",
        "-c",
        COMPETITION,
        "-f",
        str(path),
        "-m",
        message,
    ]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
