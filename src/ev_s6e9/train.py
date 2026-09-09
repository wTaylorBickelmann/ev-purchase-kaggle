"""Stratified CV, persist OOF + fold models, append EXPERIMENTS.md."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from ev_s6e9.experiments import append_chunk, format_chunk
from ev_s6e9.features import split_xy
from ev_s6e9.metrics import auc, fmt_cv, mean_std
from ev_s6e9.model import DEFAULTS, make_model, short_params
from ev_s6e9.paths import CV_JSON, OOF_CSV, OUTPUTS
from ev_s6e9.schema import ID_COL, TARGET


@dataclass
class CvResult:
    oof: np.ndarray
    fold_aucs: list[float]
    mean: float
    std: float
    models: list = field(default_factory=list)
    importances: pd.DataFrame | None = None
    params: dict = field(default_factory=dict)


def run_cv(
    df: pd.DataFrame,
    *,
    folds: int = 5,
    seed: int = 42,
    early_stopping: int = 50,
    model_overrides: dict | None = None,
) -> CvResult:
    x, y = split_xy(df)
    yv = y.to_numpy()
    oof = np.zeros(len(y), dtype=float)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    models, scores, imps = [], [], []
    overrides = dict(model_overrides or {})
    for i, (tr, va) in enumerate(skf.split(x, yv)):
        m = make_model(seed=seed + i, **overrides)
        m.fit(
            x.iloc[tr],
            yv[tr],
            eval_set=[(x.iloc[va], yv[va])],
            callbacks=[
                lgb.early_stopping(early_stopping, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        p = m.predict_proba(x.iloc[va])[:, 1]
        oof[va] = p
        scores.append(auc(yv[va], p))
        models.append(m)
        imps.append(m.feature_importances_)
    mean, std = mean_std(scores)
    imp = pd.DataFrame(imps, columns=list(x.columns)).mean(axis=0).sort_values(ascending=False)
    params = {**DEFAULTS, **overrides}
    return CvResult(oof, scores, mean, std, models, imp.to_frame("importance"), params)


def save_run(df: pd.DataFrame, cv: CvResult, out: Path | None = None) -> None:
    out = out or OUTPUTS
    out.mkdir(parents=True, exist_ok=True)
    models = out / "models"
    models.mkdir(exist_ok=True)
    pd.DataFrame({ID_COL: df[ID_COL], TARGET: cv.oof}).to_csv(out / OOF_CSV.name, index=False)
    for i, m in enumerate(cv.models):
        joblib.dump(m, models / f"fold{i}.joblib")
    if cv.importances is not None:
        cv.importances.to_csv(out / "feature_importance.csv")
    payload = {
        "fold_aucs": cv.fold_aucs,
        "mean": cv.mean,
        "std": cv.std,
        "cv": fmt_cv(cv.mean, cv.std),
        "params": {k: v for k, v in cv.params.items() if k != "random_state"},
        "n": int(len(df)),
        "folds": len(cv.fold_aucs),
    }
    (out / CV_JSON.name).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def log_experiment(
    cv: CvResult,
    *,
    folds: int,
    note: str = "",
    path: Path | None = None,
) -> Path:
    title = f"LightGBM {short_params(cv.params)}, raw+charging_total, {folds}-fold"
    chunk = format_chunk(
        title,
        fmt_cv(cv.mean, cv.std),
        takeaway=note or "auto-logged from train",
    )
    return append_chunk(chunk, path=path)


def train(
    df: pd.DataFrame,
    *,
    folds: int = 5,
    seed: int = 42,
    log: bool = True,
    note: str = "",
    experiments_path: Path | None = None,
    out: Path | None = None,
    model_overrides: dict | None = None,
) -> CvResult:
    cv = run_cv(df, folds=folds, seed=seed, model_overrides=model_overrides)
    save_run(df, cv, out=out)
    if log:
        log_experiment(cv, folds=folds, note=note, path=experiments_path)
    print(f"CV AUC: {fmt_cv(cv.mean, cv.std)}")
    print("folds:", ", ".join(f"{a:.5f}" for a in cv.fold_aucs))
    return cv
