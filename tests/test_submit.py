"""Submission schema checks."""

from __future__ import annotations

import pandas as pd
import pytest

from ev_s6e9.data import synth
from ev_s6e9.predict import write_submission
from ev_s6e9.schema import ID_COL, SUB_COLS, TARGET, check_submission


def test_check_submission_ok():
    df = pd.DataFrame({ID_COL: [668665, 668666], TARGET: [0.2, 0.3]})
    check_submission(df)


def test_check_submission_bad_cols():
    with pytest.raises(ValueError, match="columns"):
        check_submission(pd.DataFrame({ID_COL: [1], "pred": [0.1]}))


def test_check_submission_dup_ids():
    with pytest.raises(ValueError, match="duplicate"):
        check_submission(pd.DataFrame({ID_COL: [1, 1], TARGET: [0.1, 0.2]}))


def test_write_submission_shape(tmp_path):
    te = synth(8, seed=4, target=False)
    path = write_submission(te[ID_COL], [0.5] * 8, path=tmp_path / "submission.csv")
    out = pd.read_csv(path)
    assert list(out.columns) == SUB_COLS
    assert len(out) == 8
    check_submission(out)
