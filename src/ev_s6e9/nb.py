"""Notebook / WASM helpers. No LightGBM — safe in the browser."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from ev_s6e9.data import has_real_data, load_train, synth
from ev_s6e9.paths import EXPERIMENTS_MD, NOTEBOOKS_PUBLIC, TRAIN_CSV, TRAIN_SAMPLE_CSV
from ev_s6e9.schema import TRAIN_COLS, check_exact

WASM_MODULES = [
    "__init__.py",
    "schema.py",
    "features.py",
    "experiments.py",
    "paths.py",
    "data.py",
    "nb.py",
]


def in_wasm() -> bool:
    return sys.platform == "emscripten"


def read_text(url_or_path: str | Path) -> str:
    s = str(url_or_path)
    if s.startswith("http://") or s.startswith("https://"):
        from urllib.request import urlopen

        return urlopen(s).read().decode("utf-8")
    return Path(s).read_text(encoding="utf-8")


def load_nb_train(*, notebook_location=None, n_synth: int = 400) -> pd.DataFrame:
    """Full train.csv locally when present; otherwise the committed sample / synth."""
    if not in_wasm() and TRAIN_CSV.exists() and has_real_data():
        return load_train()
    if TRAIN_SAMPLE_CSV.exists():
        df = pd.read_csv(TRAIN_SAMPLE_CSV)
        check_exact(df.columns, TRAIN_COLS, "train_sample")
        return df
    if notebook_location is not None:
        url = str(Path(str(notebook_location)) / "public" / "train_sample.csv")
        df = pd.read_csv(url)
        check_exact(df.columns, TRAIN_COLS, "train_sample")
        return df
    return synth(n_synth, seed=0)


def experiments_text(*, notebook_location=None) -> str:
    for p in (EXPERIMENTS_MD, NOTEBOOKS_PUBLIC / "EXPERIMENTS.md"):
        if p.exists():
            return p.read_text(encoding="utf-8")
    if notebook_location is not None:
        return read_text(Path(str(notebook_location)) / "public" / "EXPERIMENTS.md")
    return "_No EXPERIMENTS.md found._"


def wasm_banner() -> str:
    if in_wasm():
        return (
            "> **GitHub Pages / WASM:** this notebook runs in the browser on a **sample** "
            "(~hundreds of rows), not the ~670k competition file. "
            "Full LightGBM 5-fold: local `python -m ev_s6e9 train`."
        )
    return (
        "> **Local session:** `python -m ev_s6e9 download` then this notebook can read "
        "`data/raw/train.csv`. Full CV still belongs on the CLI."
    )
