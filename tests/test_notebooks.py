"""Notebooks call the library; Pages samples match the competition header."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ev_s6e9.experiments import parse_chunks
from ev_s6e9.nb import in_wasm, load_nb_train
from ev_s6e9.paths import ROOT, TRAIN_SAMPLE_CSV
from ev_s6e9.schema import TRAIN_COLS


def test_pages_sample_header():
    df = pd.read_csv(TRAIN_SAMPLE_CSV)
    assert list(df.columns) == TRAIN_COLS
    assert 50 <= len(df) <= 2000


def test_load_nb_train_uses_sample_when_no_kaggle(monkeypatch):
    monkeypatch.setattr("ev_s6e9.nb.has_real_data", lambda: False)
    monkeypatch.setattr("ev_s6e9.nb.TRAIN_CSV", ROOT / "missing.csv")
    df = load_nb_train()
    assert list(df.columns) == TRAIN_COLS


def test_not_wasm_on_agent():
    assert in_wasm() is False


def test_parse_chunks_skips_header():
    text = (ROOT / "EXPERIMENTS.md").read_text(encoding="utf-8")
    chunks = parse_chunks(text)
    assert len(chunks) >= 1
    assert all(c.startswith("### ") for c in chunks)
    assert "pipeline seed" in chunks[0]


def test_marimo_notebooks_call_library():
    nbs = list((ROOT / "notebooks").glob("*.py"))
    assert {p.stem for p in nbs} >= {"eda", "features", "training"}
    for p in nbs:
        src = p.read_text(encoding="utf-8")
        assert "ev_s6e9" in src
        assert "import marimo" in src
