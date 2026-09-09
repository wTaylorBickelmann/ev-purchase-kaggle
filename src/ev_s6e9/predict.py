"""Average fold models → submission.csv."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ev_s6e9.features import prep_x
from ev_s6e9.paths import MODELS_DIR, SUB_CSV
from ev_s6e9.schema import ID_COL, TARGET, check_submission


def load_models(models_dir: Path | None = None) -> list:
    d = models_dir or MODELS_DIR
    paths = sorted(d.glob("fold*.joblib"))
    if not paths:
        raise FileNotFoundError(f"no fold models in {d}; run train first")
    return [joblib.load(p) for p in paths]


def predict_proba(df: pd.DataFrame, models: list) -> np.ndarray:
    x = prep_x(df)
    ps = [m.predict_proba(x)[:, 1] for m in models]
    return np.mean(ps, axis=0)


def write_submission(ids, p, path: Path | None = None) -> Path:
    path = path or SUB_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    sub = pd.DataFrame({ID_COL: ids, TARGET: p})
    check_submission(sub)
    sub.to_csv(path, index=False)
    return path


def predict(
    test: pd.DataFrame,
    *,
    models_dir: Path | None = None,
    path: Path | None = None,
) -> Path:
    models = load_models(models_dir)
    p = predict_proba(test, models)
    out = write_submission(test[ID_COL], p, path=path)
    print(f"wrote {out} ({len(p)} rows)")
    return out
