"""EDA / metric plots under reports/."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ev_s6e9.features import encode_target
from ev_s6e9.paths import OUTPUTS, REPORTS
from ev_s6e9.schema import CAT_COLS, NUM_COLS, TARGET


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_target_rate(df: pd.DataFrame, out: Path | None = None) -> Path:
    y = encode_target(df[TARGET])
    fig, ax = plt.subplots(figsize=(4, 3))
    y.value_counts(normalize=True).sort_index().plot(kind="bar", ax=ax, color="#3b7ddd")
    ax.set_ylabel("rate")
    ax.set_xlabel(TARGET)
    ax.set_title(f"target rate (pos={y.mean():.3f})")
    return _save(fig, out or REPORTS / "target_rate.png")


def plot_numeric(df: pd.DataFrame, out: Path | None = None) -> Path:
    cols = [c for c in NUM_COLS if c in df.columns][:6]
    n = len(cols)
    fig, axes = plt.subplots(2, 3, figsize=(9, 5))
    for ax, c in zip(axes.ravel(), cols):
        df[c].dropna().hist(ax=ax, bins=30, color="#3b7ddd")
        ax.set_title(c, fontsize=8)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("numeric distributions")
    return _save(fig, out or REPORTS / "numeric_dists.png")


def plot_cats(df: pd.DataFrame, out: Path | None = None) -> Path:
    cols = [c for c in CAT_COLS if c in df.columns]
    fig, axes = plt.subplots(2, 3, figsize=(9, 5))
    for ax, c in zip(axes.ravel(), cols):
        df[c].astype(str).value_counts().plot(kind="bar", ax=ax, color="#5aa")
        ax.set_title(c, fontsize=8)
        ax.tick_params(axis="x", labelrotation=30, labelsize=7)
    fig.suptitle("categorical counts")
    return _save(fig, out or REPORTS / "cat_counts.png")


def plot_target_by_cat(df: pd.DataFrame, out: Path | None = None) -> Path:
    y = encode_target(df[TARGET])
    tmp = df.assign(_y=y)
    cols = [c for c in CAT_COLS if c in df.columns]
    fig, axes = plt.subplots(2, 3, figsize=(9, 5))
    for ax, c in zip(axes.ravel(), cols):
        tmp.groupby(c, observed=False)["_y"].mean().plot(kind="bar", ax=ax, color="#3b7ddd")
        ax.set_title(c, fontsize=8)
        ax.set_ylabel("pos rate")
        ax.tick_params(axis="x", labelrotation=30, labelsize=7)
    for ax in axes.ravel()[len(cols) :]:
        ax.axis("off")
    fig.suptitle("P(Will_Buy_EV=1) by category")
    return _save(fig, out or REPORTS / "target_by_cat.png")


def plot_importance(path: Path | None = None, out: Path | None = None) -> Path | None:
    src = path or (OUTPUTS / "feature_importance.csv")
    if not src.exists():
        return None
    s = pd.read_csv(src, index_col=0).squeeze("columns")
    fig, ax = plt.subplots(figsize=(6, 4))
    s.sort_values().plot(kind="barh", ax=ax, color="#3b7ddd")
    ax.set_title("mean LightGBM gain/split importance")
    return _save(fig, out or REPORTS / "feature_importance.png")


def eda(df: pd.DataFrame, reports: Path | None = None) -> list[Path]:
    reports = reports or REPORTS
    reports.mkdir(parents=True, exist_ok=True)
    paths = [
        plot_target_rate(df, reports / "target_rate.png"),
        plot_numeric(df, reports / "numeric_dists.png"),
        plot_cats(df, reports / "cat_counts.png"),
    ]
    imp = plot_importance(out=reports / "feature_importance.png")
    if imp:
        paths.append(imp)
    for p in paths:
        print(p)
    return paths
