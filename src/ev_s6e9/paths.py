"""Repo-relative paths."""

from __future__ import annotations

import os
from pathlib import Path


def _root() -> Path:
    env = os.environ.get("EV_S6E9_ROOT")
    if env:
        return Path(env)
    here = Path(__file__).resolve().parent
    parent = here.parent
    if parent.name == "src" and (parent.parent / "src" / "ev_s6e9").exists():
        return parent.parent
    if parent.name == "src":
        return parent.parent
    return Path.cwd()


ROOT = _root()
DATA_RAW = ROOT / "data" / "raw"
OUTPUTS = ROOT / "outputs"
REPORTS = ROOT / "reports"
EXPERIMENTS_MD = ROOT / "EXPERIMENTS.md"
NOTEBOOKS_PUBLIC = ROOT / "notebooks" / "public"

TRAIN_CSV = DATA_RAW / "train.csv"
TEST_CSV = DATA_RAW / "test.csv"
SAMPLE_CSV = DATA_RAW / "sample_submission.csv"
TRAIN_SAMPLE_CSV = NOTEBOOKS_PUBLIC / "train_sample.csv"
TEST_SAMPLE_CSV = NOTEBOOKS_PUBLIC / "test_sample.csv"

OOF_CSV = OUTPUTS / "oof.csv"
SUB_CSV = OUTPUTS / "submission.csv"
CV_JSON = OUTPUTS / "cv.json"
MODELS_DIR = OUTPUTS / "models"
