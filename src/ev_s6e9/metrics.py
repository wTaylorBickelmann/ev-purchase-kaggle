"""ROC-AUC helpers."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def auc(y, p) -> float:
    return float(roc_auc_score(y, p))


def mean_std(scores: list[float]) -> tuple[float, float]:
    a = np.asarray(scores, dtype=float)
    return float(a.mean()), float(a.std(ddof=0))


def fmt_cv(mean: float, std: float) -> str:
    return f"{mean:.5f} ± {std:.5f}"
