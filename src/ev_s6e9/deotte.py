"""Deotte Fable 5.1: 3-model XGB blend (baseline, base_margin, recipe_feature)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from ev_s6e9.experiments import append_chunk, format_chunk
from ev_s6e9.features import FeatureBuilder, encode_target
from ev_s6e9.metrics import auc, fmt_cv, mean_std
from ev_s6e9.model import XGB_DEFAULTS, make_xgb_model, short_xgb_params
from ev_s6e9.paths import CV_JSON, OOF_CSV, OUTPUTS
from ev_s6e9.schema import ID_COL, TARGET

STRATEGY = "deotte"
VARIANTS = ("m1", "m2", "m3")


class DeotteVariant(Enum):
    BASELINE = "m1"
    BASE_MARGIN = "m2"
    RECIPE_FEATURE = "m3"


@dataclass
class VariantCv:
    oof: np.ndarray
    fold_aucs: list[float]
    mean: float
    std: float
    models: list = field(default_factory=list)


@dataclass
class DeotteCvResult:
    oof: np.ndarray
    fold_aucs: list[float]
    mean: float
    std: float
    variants: dict[str, VariantCv]
    feature_builder: FeatureBuilder
    params: dict = field(default_factory=dict)


def _fit_fold(
    fb: FeatureBuilder,
    variant: DeotteVariant,
    raw_tr: pd.DataFrame,
    raw_va: pd.DataFrame,
    y_tr: np.ndarray,
    y_va: np.ndarray,
    *,
    seed: int,
    overrides: dict,
) -> tuple[object, np.ndarray]:
    use_margin = variant == DeotteVariant.BASE_MARGIN
    with_recipe = variant == DeotteVariant.RECIPE_FEATURE
    x_tr = fb.transform(raw_tr, with_recipe=with_recipe)
    x_va = fb.transform(raw_va, with_recipe=with_recipe)
    m = make_xgb_model(seed=seed, **overrides)
    if use_margin:
        margin_tr = fb.recipe_logit(raw_tr)
        margin_va = fb.recipe_logit(raw_va)
        m.fit(
            x_tr,
            y_tr,
            eval_set=[(x_va, y_va)],
            base_margin=margin_tr,
            base_margin_eval_set=[margin_va],
            verbose=False,
        )
        p = m.predict_proba(x_va, base_margin=margin_va)[:, 1]
    else:
        m.fit(x_tr, y_tr, eval_set=[(x_va, y_va)], verbose=False)
        p = m.predict_proba(x_va)[:, 1]
    return m, p


def run_variant_cv(
    df: pd.DataFrame,
    fb: FeatureBuilder,
    variant: DeotteVariant,
    *,
    folds: int = 5,
    seed: int = 42,
    model_overrides: dict | None = None,
) -> VariantCv:
    y = encode_target(df[TARGET]).to_numpy()
    oof = np.zeros(len(y), dtype=float)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    models, scores = [], []
    overrides = dict(model_overrides or {})
    for i, (tr, va) in enumerate(skf.split(df, y)):
        m, p = _fit_fold(
            fb,
            variant,
            df.iloc[tr],
            df.iloc[va],
            y[tr],
            y[va],
            seed=seed + i,
            overrides=overrides,
        )
        oof[va] = p
        scores.append(auc(y[va], p))
        models.append(m)
    mean, std = mean_std(scores)
    return VariantCv(oof, scores, mean, std, models)


def run_cv(
    train: pd.DataFrame,
    test: pd.DataFrame | None = None,
    *,
    folds: int = 5,
    seed: int = 42,
    model_overrides: dict | None = None,
    freq: bool = False,
) -> DeotteCvResult:
    fb = FeatureBuilder(freq=freq).fit(train, test)
    variants: dict[str, VariantCv] = {}
    oofs = []
    for v in DeotteVariant:
        vc = run_variant_cv(train, fb, v, folds=folds, seed=seed, model_overrides=model_overrides)
        variants[v.value] = vc
        oofs.append(vc.oof)
    blend_oof = np.mean(oofs, axis=0)
    y = encode_target(train[TARGET]).to_numpy()
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    blend_scores = []
    for tr, va in skf.split(train, y):
        blend_scores.append(auc(y[va], blend_oof[va]))
    mean, std = mean_std(blend_scores)
    params = {**XGB_DEFAULTS, **(model_overrides or {})}
    return DeotteCvResult(blend_oof, blend_scores, mean, std, variants, fb, params)


def save_run(df: pd.DataFrame, cv: DeotteCvResult, out: Path | None = None) -> None:
    out = out or OUTPUTS
    out.mkdir(parents=True, exist_ok=True)
    models_root = out / "models"
    models_root.mkdir(exist_ok=True)
    for name in VARIANTS:
        (models_root / name).mkdir(exist_ok=True)
    joblib.dump(cv.feature_builder, out / "feature_builder.joblib")
    pd.DataFrame({ID_COL: df[ID_COL], TARGET: cv.oof}).to_csv(out / OOF_CSV.name, index=False)
    for name, vc in cv.variants.items():
        for i, m in enumerate(vc.models):
            joblib.dump(m, models_root / name / f"fold{i}.joblib")
    payload = {
        "strategy": STRATEGY,
        "fold_aucs": cv.fold_aucs,
        "mean": cv.mean,
        "std": cv.std,
        "cv": fmt_cv(cv.mean, cv.std),
        "params": {k: v for k, v in cv.params.items() if k != "random_state"},
        "n": int(len(df)),
        "folds": len(cv.fold_aucs),
        "variants": {
            name: {"fold_aucs": vc.fold_aucs, "mean": vc.mean, "std": vc.std, "cv": fmt_cv(vc.mean, vc.std)}
            for name, vc in cv.variants.items()
        },
    }
    (out / CV_JSON.name).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def log_experiment(
    cv: DeotteCvResult,
    *,
    folds: int,
    note: str = "",
    path: Path | None = None,
) -> Path:
    title = f"Deotte XGB 3-model blend ({short_xgb_params(cv.params)}), {folds}-fold"
    chunk = format_chunk(
        title,
        fmt_cv(cv.mean, cv.std),
        takeaway=note or "auto-logged from train --strategy deotte",
    )
    return append_chunk(chunk, path=path)


def train(
    df: pd.DataFrame,
    test: pd.DataFrame | None = None,
    *,
    folds: int = 5,
    seed: int = 42,
    log: bool = True,
    note: str = "",
    experiments_path: Path | None = None,
    out: Path | None = None,
    model_overrides: dict | None = None,
    freq: bool = False,
) -> DeotteCvResult:
    cv = run_cv(df, test, folds=folds, seed=seed, model_overrides=model_overrides, freq=freq)
    save_run(df, cv, out=out)
    if log:
        log_experiment(cv, folds=folds, note=note, path=experiments_path)
    print(f"CV AUC (blend): {fmt_cv(cv.mean, cv.std)}")
    for name in VARIANTS:
        vc = cv.variants[name]
        print(f"  {name}: {fmt_cv(vc.mean, vc.std)}")
    print("folds:", ", ".join(f"{a:.5f}" for a in cv.fold_aucs))
    return cv


def _predict_variant(
    df: pd.DataFrame,
    fb: FeatureBuilder,
    variant: str,
    models: list,
) -> np.ndarray:
    v = DeotteVariant(variant)
    use_margin = v == DeotteVariant.BASE_MARGIN
    with_recipe = v == DeotteVariant.RECIPE_FEATURE
    x = fb.transform(df, with_recipe=with_recipe)
    if use_margin:
        margin = fb.recipe_logit(df)
        ps = [m.predict_proba(x, base_margin=margin)[:, 1] for m in models]
    else:
        ps = [m.predict_proba(x)[:, 1] for m in models]
    return np.mean(ps, axis=0)


def predict_proba(df: pd.DataFrame, out: Path | None = None) -> np.ndarray:
    out = out or OUTPUTS
    fb = joblib.load(out / "feature_builder.joblib")
    preds = []
    for name in VARIANTS:
        paths = sorted((out / "models" / name).glob("fold*.joblib"))
        if not paths:
            raise FileNotFoundError(f"no fold models in {out / 'models' / name}; run train --strategy deotte")
        models = [joblib.load(p) for p in paths]
        preds.append(_predict_variant(df, fb, name, models))
    return np.mean(preds, axis=0)
