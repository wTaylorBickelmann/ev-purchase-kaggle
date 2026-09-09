# EV Purchase Prediction (Kaggle S6E9)

Python baseline for [Predicting Electric Vehicle Purchases](https://www.kaggle.com/competitions/playground-series-s6e9)
(`playground-series-s6e9`). Metric: **ROC-AUC**. Target: `Will_Buy_EV` (probability).
Submission columns: `id,Will_Buy_EV`.

## Layout

```
src/ev_s6e9/     data, features, model, train, predict, submit, viz, experiments
tests/           pure helpers + import/train smoke
data/raw/        train.csv test.csv sample_submission.csv  (gitignored)
outputs/         oof.csv models/ submission.csv cv.json     (gitignored)
reports/         target rate, distributions, importance
EXPERIMENTS.md   append-only CV / LB log (train can append)
```

## Setup

```bash
git clone <this-repo>
cd ev-purchase-kaggle
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

Or: `pip install -r requirements.txt` then `pip install -e .`

## Kaggle credentials

Download needs `~/.kaggle/kaggle.json` (from [Kaggle settings](https://www.kaggle.com/settings)):

```bash
mkdir -p ~/.kaggle
chmod 600 ~/.kaggle/kaggle.json
```

Join the competition in the browser first, or the CLI download/submit will 403.

This repo never commits credentials or competition CSVs.

## Download → train → predict → submit

```bash
python -m ev_s6e9 download
python -m ev_s6e9 train
python -m ev_s6e9 predict
python -m ev_s6e9 submit -m "lgbm 5-fold baseline"
```

Equivalent submit:

```bash
kaggle competitions submit -c playground-series-s6e9 -f outputs/submission.csv -m "lgbm 5-fold baseline"
```

`train` runs stratified 5-fold LightGBM, prints **mean AUC ± std**, writes `outputs/`,
plots under `reports/`, and **appends** a chunk to `EXPERIMENTS.md` (use `--no-log` to skip;
`--note "one-line takeaway"` to set the takeaway). Do not rewrite old experiment entries.

EDA only: `python -m ev_s6e9 eda`

## No credentials on this machine?

The pipeline still runs on schema-accurate synthetic CSVs (same column names as the
competition: `id`, `Age`, `Annual_Income_USD`, `Daily_Commute_km`, `Number_of_Cars_Owned`,
`Charging_Stations_Near_Home`, `Charging_Stations_Near_Work`, `Environmental_Concern_Level`,
`Gender`, `City_Type`, `Current_Car_Type`, `Home_Charging_Possible`, `Subsidy_Available`,
`Range_Anxiety_Level`, `Will_Buy_EV`):

```bash
python -m ev_s6e9 download --synth
python -m ev_s6e9 train --synth
python -m ev_s6e9 predict
```

`--synth` train does **not** append to `EXPERIMENTS.md`. Real leaderboard numbers need
`~/.kaggle/kaggle.json` and `python -m ev_s6e9 download` (no `--synth`).

## Tests

```bash
pytest
```

## First leaderboard submission

1. Add `kaggle.json`, join the competition.
2. `python -m ev_s6e9 download && python -m ev_s6e9 train && python -m ev_s6e9 predict`
3. Check `outputs/cv.json` (mean AUC ± std) and `outputs/submission.csv`.
4. `python -m ev_s6e9 submit -m "lgbm 5-fold baseline"`
5. Paste the public LB score into the new `EXPERIMENTS.md` chunk (`LB: —` → the score).
