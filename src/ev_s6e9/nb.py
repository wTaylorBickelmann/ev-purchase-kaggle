"""Notebook helpers for local `marimo edit`. Pages does not execute these."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ev_s6e9.data import load_train, synth
from ev_s6e9.paths import EXPERIMENTS_MD, TRAIN_CSV, TRAIN_SAMPLE_CSV
from ev_s6e9.schema import TRAIN_COLS, check_exact
from ev_s6e9.site import is_full_train


def load_nb_train(*, n_synth: int = 400) -> pd.DataFrame:
    """Full train.csv when present; otherwise the committed sample."""
    if is_full_train(TRAIN_CSV):
        return load_train()
    if TRAIN_SAMPLE_CSV.exists():
        df = pd.read_csv(TRAIN_SAMPLE_CSV)
        check_exact(df.columns, TRAIN_COLS, "train_sample")
        return df
    return synth(n_synth, seed=0)


def experiments_text() -> str:
    p = EXPERIMENTS_MD
    return p.read_text(encoding="utf-8") if p.exists() else "_No EXPERIMENTS.md found._"


def data_banner() -> str:
    if is_full_train(TRAIN_CSV):
        return "> Using `data/raw/train.csv` (full competition file)."
    return (
        "> No full train.csv — using the committed sample. "
        "Download with `python -m ev_s6e9 download`. Full CV: `python -m ev_s6e9 train`."
    )
