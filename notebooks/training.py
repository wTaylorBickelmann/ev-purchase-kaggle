# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "pandas",
#     "numpy",
#     "altair",
# ]
# ///
"""Review EXPERIMENTS.md + optional tiny local train. Full CV is CLI-only."""

import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
async def _():
    import sys
    from pathlib import Path
    import marimo as mo

    nd = Path(str(mo.notebook_dir()))
    for p in (nd.parent / "src", Path.cwd() / "src"):
        if (p / "ev_s6e9").is_dir() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
            break
    try:
        import ev_s6e9.schema  # noqa: F401
    except ImportError:
        if sys.platform != "emscripten":
            raise
        from pyodide.http import pyfetch

        dest = Path("/session/src/ev_s6e9")
        dest.mkdir(parents=True, exist_ok=True)
        loc = mo.notebook_location()
        for name in (
            "__init__.py",
            "schema.py",
            "features.py",
            "experiments.py",
            "paths.py",
            "data.py",
            "nb.py",
        ):
            r = await pyfetch(str(loc / "public" / "ev_s6e9" / name))
            (dest / name).write_bytes(await r.bytes())
        sys.path.insert(0, "/session/src")
        import ev_s6e9.schema  # noqa: F401
    return (mo,)


@app.cell
def _(mo):
    from ev_s6e9.experiments import parse_chunks
    from ev_s6e9.nb import experiments_text, in_wasm, load_nb_train, wasm_banner

    loc = mo.notebook_location()
    log = experiments_text(notebook_location=loc)
    chunks = parse_chunks(log)
    mo.md(
        f"""
        # Training & experiments

        {wasm_banner()}

        `EXPERIMENTS.md` is **append-only**. Local `python -m ev_s6e9 train` appends a chunk
        (CV, model, key params). Do not rewrite old entries.

        Full LightGBM 5-fold on ~670k rows is **not** run in WASM.
        """
    )
    return chunks, in_wasm, load_nb_train, loc, log


@app.cell
def _(log, mo):
    mo.md(log)
    return


@app.cell
def _(chunks, mo):
    mo.md(f"**{len(chunks)}** logged chunk(s). Newest at the bottom.")
    return


@app.cell
def _(in_wasm, mo):
    import pandas as pd

    if in_wasm():
        btn = None
        mo.md("WASM: use the CLI on your machine for a real CV score.")
    else:
        btn = mo.ui.run_button(label="Tiny local sample train (2-fold, not the leaderboard run)")
        btn
    return btn, pd


@app.cell
def _(btn, in_wasm, load_nb_train, loc, mo):
    if in_wasm() or btn is None or not btn.value:
        out = mo.md("_No local train this run._")
    else:
        from ev_s6e9.train import train

        df = load_nb_train(notebook_location=loc)
        cv = train(
            df,
            folds=2,
            seed=0,
            log=False,
            model_overrides={"n_estimators": 40},
        )
        out = mo.md(
            f"Sample CV (not EXPERIMENTS.md, not LB): **{cv.mean:.4f} ± {cv.std:.4f}**. "
            f"Leaderboard path: `python -m ev_s6e9 download && python -m ev_s6e9 train && "
            f"python -m ev_s6e9 predict && python -m ev_s6e9 submit -m 'lgbm baseline'`"
        )
    out
    return


@app.cell
def _(loc, mo, pd):
    from pathlib import Path

    url = str(Path(str(loc)) / "public" / "feature_importance.csv")
    try:
        imp = pd.read_csv(url, index_col=0)
        imp
    except Exception:
        mo.md("_No feature_importance.csv in public/ yet._")
    return


if __name__ == "__main__":
    app.run()
