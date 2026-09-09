# EV purchase notebooks (Kaggle S6E9)

Marimo-first workspace for [Predicting Electric Vehicle Purchases](https://www.kaggle.com/competitions/playground-series-s6e9)
(`playground-series-s6e9`, ROC-AUC, submit `id,Will_Buy_EV`).

**Primary UX:** interactive marimo notebooks (EDA, features, training log) on **GitHub Pages**.
`src/ev_s6e9/` is the small library those notebooks (and the optional CLI) call.

## GitHub Pages

After merge to `main`:

1. Repo **Settings → Pages → Source: GitHub Actions**
2. Push to `main` (or **Actions → Deploy to GitHub Pages → Run workflow**)

Site:

- https://wTaylorBickelmann.github.io/ev-purchase-kaggle/
- https://wTaylorBickelmann.github.io/ev-purchase-kaggle/notebooks/eda.html
- https://wTaylorBickelmann.github.io/ev-purchase-kaggle/notebooks/features.html
- https://wTaylorBickelmann.github.io/ev-purchase-kaggle/notebooks/training.html

Export follows the [official marimo GitHub Pages template](https://github.com/marimo-team/marimo-gh-pages-template):
`marimo export html-wasm … --mode edit` for `notebooks/` (see `.github/scripts/build.py` and `.github/workflows/deploy.yml`).
The build writes `.nojekyll` and an index that links the notebooks.

### WASM vs local (important)

| | GitHub Pages (Pyodide / WASM) | Your machine |
|---|---|---|
| Data | committed `notebooks/public/*_sample.csv` (hundreds of rows) | `python -m ev_s6e9 download` → `data/raw/` (~670k train) |
| Charts / EXPERIMENTS.md | yes | yes |
| LightGBM 5-fold | **no** | `python -m ev_s6e9 train` |
| Kaggle credentials | **never** | `~/.kaggle/kaggle.json` for download/submit |

Pages must not ship full `train.csv` or secrets.

## Run notebooks locally

```bash
git clone https://github.com/wTaylorBickelmann/ev-purchase-kaggle
cd ev-purchase-kaggle
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev,nb]"
```

```bash
marimo edit notebooks/eda.py
marimo edit notebooks/features.py
marimo edit notebooks/training.py
```

Or all: `marimo edit notebooks/`

Local export smoke (same script as CI):

```bash
uv run .github/scripts/build.py
python -m http.server -d _site
```

## Local train / predict / submit (CLI)

Still the way to make a leaderboard submission:

```bash
# ~/.kaggle/kaggle.json  (chmod 600) + join the competition in the browser
python -m ev_s6e9 download
python -m ev_s6e9 train          # mean AUC ± std; appends EXPERIMENTS.md
python -m ev_s6e9 predict        # outputs/submission.csv
python -m ev_s6e9 submit -m "lgbm 5-fold baseline"
```

`train` is append-only on `EXPERIMENTS.md` (`--no-log` to skip; `--note "…"` for the takeaway).
Do not edit old experiment chunks.

No credentials? Schema-accurate fake CSVs (same column names, not for LB):

```bash
python -m ev_s6e9 download --synth
python -m ev_s6e9 train --synth
python -m ev_s6e9 predict
```

## Layout

```
notebooks/           marimo apps (thin; call the library)
notebooks/public/    sample CSVs + staged WASM copy of ev_s6e9 (CI)
src/ev_s6e9/         data, features, model, train, predict, submit, viz, experiments, nb
.github/scripts/build.py    official-template html-wasm export + index
.github/workflows/deploy.yml
EXPERIMENTS.md       append-only CV / LB log
```

Columns (do not invent others): `id, Age, Annual_Income_USD, Daily_Commute_km, Number_of_Cars_Owned, Charging_Stations_Near_Home, Charging_Stations_Near_Work, Environmental_Concern_Level, Gender, City_Type, Current_Car_Type, Home_Charging_Possible, Subsidy_Available, Range_Anxiety_Level, Will_Buy_EV`.

## Tests

```bash
pytest
```
