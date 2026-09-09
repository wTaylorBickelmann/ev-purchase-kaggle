"""LightGBM factory. Small, readable defaults for a first AUC baseline."""

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


def lgbm_params(**overrides: Any) -> dict[str, Any]:
    p = dict(DEFAULTS)
    p.update(overrides)
    return p


def make_model(seed: int = 42, **overrides: Any) -> lgb.LGBMClassifier:
    p = lgbm_params(random_state=seed, **overrides)
    return lgb.LGBMClassifier(**p)


def short_params(p: dict[str, Any] | None = None) -> str:
    """One-line key params for EXPERIMENTS.md."""
    p = p or DEFAULTS
    return (
        f"n_estimators={p.get('n_estimators')} lr={p.get('learning_rate')} "
        f"num_leaves={p.get('num_leaves')}"
    )
