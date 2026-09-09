# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "pandas",
#     "numpy",
#     "altair",
# ]
# ///
"""Feature inspection. Calls src/ev_s6e9.features."""

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
    from ev_s6e9.features import split_xy
    from ev_s6e9.nb import load_nb_train, wasm_banner
    from ev_s6e9.schema import CAT_COLS, FEATURE_COLS

    loc = mo.notebook_location()
    raw = load_nb_train(notebook_location=loc)
    x, y = split_xy(raw)
    mo.md(
        f"""
        # Data engineering

        {wasm_banner()}

        `prep_x` / `split_xy` live in `src/ev_s6e9/features.py` (numeric coerce, categoricals, `charging_total`).
        """
    )
    return CAT_COLS, FEATURE_COLS, loc, raw, x, y


@app.cell
def _(FEATURE_COLS, mo, raw, x):
    mo.md(
        f"""
        - raw cols: {len(raw.columns)} — `{list(raw.columns)}`
        - model matrix: {x.shape} — extra: `{[c for c in x.columns if c not in FEATURE_COLS]}`
        """
    )
    return


@app.cell
def _(x):
    x.dtypes.astype(str).rename("dtype")
    return


@app.cell
def _(x):
    x[["Charging_Stations_Near_Home", "Charging_Stations_Near_Work", "charging_total"]].head()
    return


@app.cell
def _(CAT_COLS, mo, x, y):
    import altair as alt

    tmp = x.assign(Will_Buy_EV=y.values)
    col = mo.ui.dropdown(options=list(CAT_COLS), value="Range_Anxiety_Level", label="vs target")
    col
    return alt, col, tmp


@app.cell
def _(alt, col, mo, tmp):
    g = (
        tmp.groupby(col.value, observed=False)["Will_Buy_EV"]
        .mean()
        .rename("pos_rate")
        .reset_index()
    )
    mo.ui.altair_chart(
        alt.Chart(g)
        .mark_bar()
        .encode(x=f"{col.value}:N", y="pos_rate:Q")
        .properties(title=f"P(Will_Buy_EV=1) by {col.value}", width=360, height=220),
        chart_selection=None,
    )
    return


if __name__ == "__main__":
    app.run()
