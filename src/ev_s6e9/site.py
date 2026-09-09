"""Static read-only GitHub Pages gallery (executed figures + experiment log)."""

from __future__ import annotations

import html
import json
import shutil
from pathlib import Path

import pandas as pd

from ev_s6e9.data import load_train
from ev_s6e9.experiments import parse_chunks
from ev_s6e9.features import split_xy
from ev_s6e9.paths import (
    CV_JSON,
    EXPERIMENTS_MD,
    OUTPUTS,
    ROOT,
    TRAIN_CSV,
    TRAIN_SAMPLE_CSV,
)
from ev_s6e9.schema import TRAIN_COLS, check_exact
from ev_s6e9.viz import eda, plot_importance, plot_target_by_cat

FULL_MIN_ROWS = 100_000


def is_full_train(path: Path | None = None) -> bool:
    """True only for a competition-sized train.csv (~668k), not the tiny sample."""
    p = path or TRAIN_CSV
    if not p.exists():
        return False
    n = 0
    with p.open(encoding="utf-8") as f:
        next(f, None)
        for n, _ in enumerate(f, 1):
            if n >= FULL_MIN_ROWS:
                return True
    return False


def frame_for_site(*, sample: bool = False) -> tuple[pd.DataFrame, str]:
    """Prefer full `data/raw/train.csv`; otherwise committed Pages sample."""
    if not sample and is_full_train(TRAIN_CSV):
        return load_train(), f"data/raw/train.csv ({FULL_MIN_ROWS}+ rows)"
    if TRAIN_SAMPLE_CSV.exists():
        df = pd.read_csv(TRAIN_SAMPLE_CSV)
        check_exact(df.columns, TRAIN_COLS, "train_sample")
        return df, f"notebooks/public/train_sample.csv ({len(df)} rows)"
    from ev_s6e9.data import synth

    df = synth(400, seed=0)
    return df, f"synthetic smoke ({len(df)} rows)"


def _page(title: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: Georgia, serif; max-width: 880px; margin: 2rem auto; padding: 0 1rem; line-height: 1.45; color: #222; }}
    nav a {{ margin-right: 1rem; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
    pre, table {{ font-family: ui-monospace, monospace; font-size: 0.85rem; overflow-x: auto; }}
    pre {{ background: #f6f6f6; padding: 1rem; white-space: pre-wrap; }}
    table {{ border-collapse: collapse; }}
    td, th {{ border: 1px solid #ccc; padding: 0.25rem 0.5rem; text-align: left; }}
    .note {{ background: #fff8e5; padding: 0.75rem 1rem; border: 1px solid #e6d9a8; }}
    footer {{ margin-top: 2rem; color: #666; font-size: 0.9rem; }}
  </style>
</head>
<body>
<nav>
  <a href="index.html">Gallery</a>
  <a href="eda.html">EDA</a>
  <a href="features.html">Features</a>
  <a href="training.html">Training</a>
</nav>
{body}
<footer>Read-only reports. New experiments: local <code>marimo edit</code> / <code>python -m ev_s6e9 train</code>.</footer>
</body>
</html>
"""


def _img(rel: str, cap: str) -> str:
    return f'<figure><img src="{html.escape(rel)}" alt="{html.escape(cap)}"><figcaption>{html.escape(cap)}</figcaption></figure>'


def _table(df: pd.DataFrame, n: int = 8) -> str:
    return df.head(n).to_html(index=False, escape=True)


def build_site(out: Path | None = None, *, sample: bool = False) -> Path:
    """Write static HTML + PNGs under `_site/` for GitHub Pages."""
    out = Path(out) if out else ROOT / "_site"
    if out.exists():
        shutil.rmtree(out)
    img = out / "reports"
    img.mkdir(parents=True)
    (out / ".nojekyll").write_text("", encoding="utf-8")

    df, source = frame_for_site(sample=sample)
    n = len(df)
    full = is_full_train() and not sample
    note = (
        f"Figures from <strong>{html.escape(source)}</strong>."
        if full
        else (
            f"EDA figures from <strong>{html.escape(source)}</strong> — not the ~670k Kaggle file. "
            "CV / LB below come from <code>EXPERIMENTS.md</code> (logged after local/full train). "
            "Rebuild with full data: <code>python -m ev_s6e9 download && python -m ev_s6e9 build_site</code>."
        )
    )

    eda(df, reports=img)
    plot_target_by_cat(df, img / "target_by_cat.png")
    imp_src = OUTPUTS / "feature_importance.csv"
    if not imp_src.exists():
        pub = ROOT / "notebooks" / "public" / "feature_importance.csv"
        imp_src = pub if pub.exists() else None
    if imp_src and Path(imp_src).exists():
        plot_importance(path=Path(imp_src), out=img / "feature_importance.png")

    x, y = split_xy(df)
    derived = x[["Charging_Stations_Near_Home", "Charging_Stations_Near_Work", "charging_total"]].describe()
    pos = (
        df.assign(Will_Buy_EV=y.values)
        .groupby("Range_Anxiety_Level", observed=False)["Will_Buy_EV"]
        .mean()
        .rename("pos_rate")
        .reset_index()
    )

    log = EXPERIMENTS_MD.read_text(encoding="utf-8") if EXPERIMENTS_MD.exists() else ""
    chunks = parse_chunks(log)
    cv_txt = ""
    if CV_JSON.exists():
        cv_txt = json.dumps(json.loads(CV_JSON.read_text(encoding="utf-8")), indent=2)

    (out / "index.html").write_text(
        _page(
            "S6E9 report gallery",
            f"""
<h1>Predicting Electric Vehicle Purchases</h1>
<p class="note">{note}</p>
<p>Kaggle <a href="https://www.kaggle.com/competitions/playground-series-s6e9">playground-series-s6e9</a>
· metric ROC-AUC · target <code>Will_Buy_EV</code>. This site is a <strong>read-only gallery</strong>
of already-run EDA / features / training — visitors do not download data or fit models.</p>
<ul>
  <li><a href="eda.html">EDA</a> — target rate, distributions ({n:,} rows in the plot frame)</li>
  <li><a href="features.html">Data engineering</a> — <code>prep_x</code>, <code>charging_total</code>, target by category</li>
  <li><a href="training.html">Training</a> — EXPERIMENTS.md, CV / LB</li>
</ul>
<p>Latest experiment chunk:</p>
<pre>{html.escape(chunks[-1] if chunks else "(none yet)")}</pre>
""",
        ),
        encoding="utf-8",
    )

    (out / "eda.html").write_text(
        _page(
            "EDA",
            f"""
<h1>EDA</h1>
<p class="note">{note}</p>
<p>Competition header: <code>{html.escape(", ".join(TRAIN_COLS))}</code></p>
{_img("reports/target_rate.png", "Target rate")}
{_img("reports/numeric_dists.png", "Numeric distributions")}
{_img("reports/cat_counts.png", "Categorical counts")}
<h2>Sample rows</h2>
{_table(df)}
""",
        ),
        encoding="utf-8",
    )

    imp_html = (
        _img("reports/feature_importance.png", "Feature importance")
        if (img / "feature_importance.png").exists()
        else "<p>No feature_importance.png (run local <code>train</code> to write outputs/).</p>"
    )
    (out / "features.html").write_text(
        _page(
            "Features",
            f"""
<h1>Data engineering</h1>
<p class="note">{note}</p>
<p><code>prep_x</code> / <code>add_derived</code> in <code>src/ev_s6e9/features.py</code>:
numeric coerce, categoricals, <code>charging_total</code> = home + work chargers.
Model matrix shape {x.shape}.</p>
{_img("reports/target_by_cat.png", "Positive rate by categorical")}
{imp_html}
<h2>charging_total</h2>
{derived.to_html(escape=True)}
<h2>P(Will_Buy_EV=1) by Range_Anxiety_Level</h2>
{pos.to_html(index=False, escape=True)}
<h2>Feature dtypes after prep_x</h2>
{x.dtypes.astype(str).rename("dtype").to_frame().to_html(escape=True)}
""",
        ),
        encoding="utf-8",
    )

    cv_block = f"<h2>outputs/cv.json</h2><pre>{html.escape(cv_txt)}</pre>" if cv_txt else ""
    (out / "training.html").write_text(
        _page(
            "Training",
            f"""
<h1>Training &amp; experiments</h1>
<p class="note">Append-only log. Newest at the bottom. Local <code>python -m ev_s6e9 train</code> appends a chunk.
Pages never runs LightGBM.</p>
{cv_block}
<h2>EXPERIMENTS.md</h2>
<pre>{html.escape(log)}</pre>
""",
        ),
        encoding="utf-8",
    )
    print(f"wrote {out} ({n} rows, source={source})")
    return out
