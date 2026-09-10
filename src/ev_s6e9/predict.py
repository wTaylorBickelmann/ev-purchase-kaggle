"""Average fold models → submission.csv."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ev_s6e9.features import prep_x
from ev_s6e9.paths import CV_JSON, MODELS_DIR, OUTPUTS, SUB_CSV
from ev_s6e9.schema import ID_COL, TARGET, check_submission


def load_strategy(out: Path | None = None) -> str:
    path = (out or OUTPUTS) / CV_JSON.name
    if not path.exists():
        return "lgbm"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("strategy", "lgbm")


def load_models(models_dir: Path | None = None) -> list:
    d = models_dir or MODELS_DIR
    paths = sorted(d.glob("fold*.joblib"))
    if not paths:
        raise FileNotFoundError(f"no fold models in {d}; run train first")
    return [joblib.load(p) for p in paths]


def predict_proba_lgbm(df: pd.DataFrame, models: list) -> np.ndarray:
    x = prep_x(df)
    ps = [m.predict_proba(x)[:, 1] for m in models]
    return np.mean(ps, axis=0)


def predict_proba(df: pd.DataFrame, *, strategy: str | None = None, out: Path | None = None) -> np.ndarray:
    out = out or OUTPUTS
    strategy = strategy or load_strategy(out)
    if strategy == "deotte":
        from ev_s6e9.deotte import predict_proba as deotte_proba

        return deotte_proba(df, out=out)
    return predict_proba_lgbm(df, load_models(out / "models"))


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
    strategy: str | None = None,
    models_dir: Path | None = None,
    path: Path | None = None,
    out: Path | None = None,
) -> Path:
    out = out or (models_dir.parent if models_dir else OUTPUTS)
    p = predict_proba(test, strategy=strategy, out=out)
    out_path = write_submission(test[ID_COL], p, path=path)
    print(f"wrote {out_path} ({len(p)} rows)")
    return out_path
