# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "pandas",
#     "numpy",
#     "altair",
# ]
# ///
"""EDA for playground-series-s6e9. Logic lives in src/ev_s6e9/."""

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
    from ev_s6e9.features import encode_target
    from ev_s6e9.nb import load_nb_train, wasm_banner
    from ev_s6e9.schema import CAT_COLS, NUM_COLS, TARGET, TRAIN_COLS

    loc = mo.notebook_location()
    df = load_nb_train(notebook_location=loc)
    mo.md(
        f"""
        # EDA — Will_Buy_EV

        {wasm_banner()}

        Columns (locked): `{", ".join(TRAIN_COLS)}`

        Loaded **{len(df):,}** rows.
        """
    )
    return CAT_COLS, NUM_COLS, TARGET, df, encode_target, loc


@app.cell
def _(TARGET, df, encode_target, mo):
    import altair as alt

    y = encode_target(df[TARGET])
    rates = (
        y.value_counts(normalize=True)
        .rename("rate")
        .rename_axis("Will_Buy_EV")
        .reset_index()
    )
    chart = (
        alt.Chart(rates)
        .mark_bar()
        .encode(x="Will_Buy_EV:N", y="rate:Q", color="Will_Buy_EV:N")
        .properties(title="Target rate", width=280, height=200)
    )
    mo.ui.altair_chart(chart, chart_selection=None)
    return alt, y


@app.cell
def _(NUM_COLS, df, mo):
    mo.md("### Sample + numeric describe")
    return


@app.cell
def _(NUM_COLS, df):
    df.head()
    return


@app.cell
def _(NUM_COLS, df):
    df[NUM_COLS].describe()
    return


@app.cell
def _(CAT_COLS, alt, df, mo):
    c = mo.ui.dropdown(options=list(CAT_COLS), value="City_Type", label="categorical")
    c
    return (c,)


@app.cell
def _(alt, c, df, mo):
    vc = df[c.value].astype(str).value_counts().rename("n").rename_axis(c.value).reset_index()
    mo.ui.altair_chart(
        alt.Chart(vc)
        .mark_bar()
        .encode(x=f"{c.value}:N", y="n:Q")
        .properties(title=c.value, width=360, height=220),
        chart_selection=None,
    )
    return


if __name__ == "__main__":
    app.run()
