"""Import smoke + tiny train/predict on synthetic frames."""

from __future__ import annotations

import pandas as pd

import ev_s6e9
from ev_s6e9.data import synth
from ev_s6e9.predict import predict
from ev_s6e9.schema import SUB_COLS, TRAIN_COLS
from ev_s6e9.train import train


def test_package_imports():
    assert ev_s6e9.__version__
    import ev_s6e9.data as data
    import ev_s6e9.experiments as experiments
    import ev_s6e9.features as features
    import ev_s6e9.metrics as metrics
    import ev_s6e9.model as model
    import ev_s6e9.predict as predict_mod
    import ev_s6e9.submit as submit
    import ev_s6e9.train as train_mod
    import ev_s6e9.viz as viz

    assert data and features and model and metrics and train_mod
    assert predict_mod and submit and viz and experiments


def test_synth_schema():
    tr = synth(12, seed=0, target=True)
    te = synth(5, seed=1, target=False)
    assert list(tr.columns) == TRAIN_COLS
    assert list(te.columns) == [c for c in TRAIN_COLS if c != "Will_Buy_EV"]


def test_train_predict_smoke(tmp_path):
    tr = synth(120, seed=0, target=True)
    te = synth(20, seed=1, target=False, start_id=10_000)
    log = tmp_path / "EXPERIMENTS.md"
    out = tmp_path / "outputs"
    cv = train(
        tr,
        folds=2,
        seed=0,
        log=True,
        note="smoke",
        experiments_path=log,
        out=out,
        model_overrides={"n_estimators": 20},
    )
    assert 0.5 <= cv.mean <= 1.0
    assert "LightGBM" in log.read_text()
    assert "0." in log.read_text()
    sub = predict(te, models_dir=out / "models", path=tmp_path / "submission.csv")
    df = pd.read_csv(sub)
    assert list(df.columns) == SUB_COLS
    assert len(df) == 20
    assert df["Will_Buy_EV"].between(0, 1).all()
