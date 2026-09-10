"""LightGBM + XGBoost model factories."""

from __future__ import annotations

from typing import Any

import lightgbm as lgb

DEFAULTS: dict[str, Any] = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "n_estimators": 800,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 50,
    "verbosity": -1,
    "n_jobs": -1,
}

XGB_DEFAULTS: dict[str, Any] = {
    "objective": "binary:logistic",
    "n_estimators": 3000,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "device": "cpu",
    "eval_metric": "auc",
    "early_stopping_rounds": 100,
    "verbosity": 0,
    "n_jobs": -1,
}


def lgbm_params(**overrides: Any) -> dict[str, Any]:
    p = dict(DEFAULTS)
    p.update(overrides)
    return p


def make_model(seed: int = 42, **overrides: Any) -> lgb.LGBMClassifier:
    p = lgbm_params(random_state=seed, **overrides)
    return lgb.LGBMClassifier(**p)


def xgb_params(**overrides: Any) -> dict[str, Any]:
    p = dict(XGB_DEFAULTS)
    p.update(overrides)
    return p


def make_xgb_model(seed: int = 42, **overrides: Any) -> Any:
    import xgboost as xgb

    p = xgb_params(random_state=seed, **overrides)
    return xgb.XGBClassifier(**p)


def short_params(p: dict[str, Any] | None = None) -> str:
    """One-line key params for EXPERIMENTS.md."""
    p = p or DEFAULTS
    return (
        f"n_estimators={p.get('n_estimators')} lr={p.get('learning_rate')} "
        f"num_leaves={p.get('num_leaves')}"
    )


def short_xgb_params(p: dict[str, Any] | None = None) -> str:
    p = p or XGB_DEFAULTS
    return (
        f"n_estimators={p.get('n_estimators')} lr={p.get('learning_rate')} "
        f"max_depth={p.get('max_depth')}"
    )
