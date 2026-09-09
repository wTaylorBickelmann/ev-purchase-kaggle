# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo", "pandas", "numpy", "altair"]
# ///
"""Local feature inspection. Calls src/ev_s6e9.features."""

import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from ev_s6e9.features import split_xy
    from ev_s6e9.nb import data_banner, load_nb_train
    from ev_s6e9.schema import CAT_COLS, FEATURE_COLS

    raw = load_nb_train()
    x, y = split_xy(raw)
    mo.md(
        f"""
        # Data engineering

        {data_banner()}

        `prep_x` / `split_xy` in `src/ev_s6e9/features.py` (categoricals + `charging_total`).
        """
    )
    return CAT_COLS, FEATURE_COLS, mo, raw, x, y


@app.cell
def _(FEATURE_COLS, mo, raw, x):
    mo.md(
        f"""
        - raw cols: {len(raw.columns)}
        - model matrix: {x.shape} — extra `{[c for c in x.columns if c not in FEATURE_COLS]}`
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
