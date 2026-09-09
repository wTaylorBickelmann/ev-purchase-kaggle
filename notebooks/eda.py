# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo", "pandas", "numpy", "altair"]
# ///
"""Local EDA. Pages gallery is `python -m ev_s6e9 build_site`, not this file."""

import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from ev_s6e9.features import encode_target
    from ev_s6e9.nb import data_banner, load_nb_train
    from ev_s6e9.schema import CAT_COLS, NUM_COLS, TARGET, TRAIN_COLS

    df = load_nb_train()
    mo.md(
        f"""
        # EDA — Will_Buy_EV

        {data_banner()}

        Columns: `{", ".join(TRAIN_COLS)}` · **{len(df):,}** rows.
        """
    )
    return CAT_COLS, NUM_COLS, TARGET, df, encode_target, mo


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
    mo.ui.altair_chart(
        alt.Chart(rates)
        .mark_bar()
        .encode(x="Will_Buy_EV:N", y="rate:Q")
        .properties(title="Target rate", width=280, height=200),
        chart_selection=None,
    )
    return alt,


@app.cell
def _(df):
    df.head()
    return


@app.cell
def _(NUM_COLS, df):
    df[NUM_COLS].describe()
    return


@app.cell
def _(CAT_COLS, mo):
    c = mo.ui.dropdown(options=list(CAT_COLS), value="City_Type", label="categorical")
    c
    return (c,)


@app.cell
def _(alt, c, df, mo):
    vc = df[c.value].astype(str).value_counts().rename("n").rename_axis(c.value).reset_index()
    mo.ui.altair_chart(
        alt.Chart(vc).mark_bar().encode(x=f"{c.value}:N", y="n:Q").properties(width=360, height=220),
        chart_selection=None,
    )
    return


if __name__ == "__main__":
    app.run()
