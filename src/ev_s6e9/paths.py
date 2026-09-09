"""Repo-relative paths."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
OUTPUTS = ROOT / "outputs"
REPORTS = ROOT / "reports"
EXPERIMENTS_MD = ROOT / "EXPERIMENTS.md"

TRAIN_CSV = DATA_RAW / "train.csv"
TEST_CSV = DATA_RAW / "test.csv"
SAMPLE_CSV = DATA_RAW / "sample_submission.csv"

OOF_CSV = OUTPUTS / "oof.csv"
SUB_CSV = OUTPUTS / "submission.csv"
CV_JSON = OUTPUTS / "cv.json"
MODELS_DIR = OUTPUTS / "models"
