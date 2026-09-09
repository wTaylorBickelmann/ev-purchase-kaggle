# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo", "pandas", "numpy"]
# ///
"""Local experiment review. Full CV is CLI-only; Pages is a static gallery."""

import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from ev_s6e9.experiments import parse_chunks
    from ev_s6e9.nb import data_banner, experiments_text, load_nb_train
    from ev_s6e9.site import is_full_train

    log = experiments_text()
    chunks = parse_chunks(log)
    mo.md(
        f"""
        # Training & experiments

        {data_banner()}

        `EXPERIMENTS.md` is append-only. `python -m ev_s6e9 train` appends a chunk.
        GitHub Pages shows this log as a **static** report — it does not fit LightGBM.
        """
    )
    return chunks, is_full_train, load_nb_train, log, mo


@app.cell
def _(log, mo):
    mo.md(log)
    return


@app.cell
def _(chunks, mo):
    mo.md(f"**{len(chunks)}** logged chunk(s). Newest at the bottom.")
    return


@app.cell
def _(is_full_train, mo):
    if is_full_train():
        btn = None
        mo.md("Full train.csv present. For a leaderboard CV run the CLI, not this notebook.")
    else:
        btn = mo.ui.run_button(label="Tiny sample train (2-fold, not LB, not EXPERIMENTS.md)")
        btn
    return (btn,)


@app.cell
def _(btn, load_nb_train, mo):
    if btn is None or not btn.value:
        out = mo.md("_CLI: `python -m ev_s6e9 download && python -m ev_s6e9 train && python -m ev_s6e9 predict && python -m ev_s6e9 submit -m 'lgbm baseline'`_")
    else:
        from ev_s6e9.train import train

        cv = train(
            load_nb_train(),
            folds=2,
            seed=0,
            log=False,
            model_overrides={"n_estimators": 40},
        )
        out = mo.md(f"Sample CV (not logged): **{cv.mean:.4f} ± {cv.std:.4f}**")
    out
    return


if __name__ == "__main__":
    app.run()
