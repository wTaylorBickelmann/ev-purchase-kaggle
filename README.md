# EV purchase reports (Kaggle S6E9)

[Predicting Electric Vehicle Purchases](https://www.kaggle.com/competitions/playground-series-s6e9)
(`playground-series-s6e9`, ROC-AUC, submit `id,Will_Buy_EV`).

**GitHub Pages is a read-only gallery** of already-run EDA, feature notes, and training
metrics. Visitors do not execute notebooks, download ~670k rows, or fit LightGBM.

Local `marimo edit` + `python -m ev_s6e9 train` are how you run new experiments.

## GitHub Pages

1. Repo **Settings → Pages → Source: GitHub Actions**
2. Push `main` (or **Actions → Deploy to GitHub Pages → Run workflow**)

Site:

- https://wTaylorBickelmann.github.io/ev-purchase-kaggle/
- https://wTaylorBickelmann.github.io/ev-purchase-kaggle/eda.html
- https://wTaylorBickelmann.github.io/ev-purchase-kaggle/features.html
- https://wTaylorBickelmann.github.io/ev-purchase-kaggle/training.html

CI runs `python -m ev_s6e9 build_site` (static HTML + PNGs → `_site/`, plus `.nojekyll`).
It is **not** `marimo export html-wasm`.

Without Kaggle secrets, EDA charts use the committed sample CSVs
(`notebooks/public/train_sample.csv`). **CV / LB come from `EXPERIMENTS.md`**
(logged after a real local train). Optional repo secrets `KAGGLE_USERNAME` +
`KAGGLE_KEY` make Actions download full `train.csv` so the figures match ~670k rows.

Full-data gallery on a laptop (does not commit CSVs):

```bash
python -m ev_s6e9 download
python -m ev_s6e9 train          # appends EXPERIMENTS.md
python -m ev_s6e9 build_site     # writes _site/
python -m http.server -d _site
```

Then push the updated `EXPERIMENTS.md` (and optionally keep `_site/` local-only;
CI rebuilds the gallery on the next `main` push).

## Local notebooks (new work, not Pages)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev,nb]"
marimo edit notebooks/eda.py
marimo edit notebooks/features.py
marimo edit notebooks/training.py
```

## Local train / predict / submit

```bash
# ~/.kaggle/kaggle.json  (chmod 600) + join the competition
python -m ev_s6e9 download
python -m ev_s6e9 train
python -m ev_s6e9 predict
python -m ev_s6e9 submit -m "lgbm 5-fold baseline"
```

Train strategies (LightGBM baseline + Chris Deotte Fable 5.1 XGB blend) are listed in **`STRATEGIES.md`**.

```bash
python -m ev_s6e9 train --strategy deotte
python -m ev_s6e9 predict --strategy deotte   # or auto-detect from outputs/cv.json
```

`train` appends `EXPERIMENTS.md` (`--no-log` to skip). Do not rewrite old chunks.

## Autonomous local loop (no Cursor tokens)

Overnight CV-gated experiment loop using **Ollama + Qwen Code CLI** (default model `deepseek-r1:70b`). Submits to Kaggle only when local CV beats the prior best.

See **`LOOP.md`**. Quick start:

```bash
ollama pull deepseek-r1:70b
bash scripts/setup_loop_model.sh
python scripts/autoloop.py --dry-run
python scripts/autoloop.py --max-iters 20 --submit --push
```

No credentials? Schema-accurate fake CSVs (not for LB):

```bash
python -m ev_s6e9 download --synth
python -m ev_s6e9 train --synth
python -m ev_s6e9 train --synth --strategy deotte
python -m ev_s6e9 predict
python -m ev_s6e9 build_site --sample
```

## Layout

```
notebooks/           local marimo (thin; call the library)
notebooks/public/    small CSVs for sample EDA when data/raw is absent
src/ev_s6e9/         data, features, model, deotte, train, predict, submit, viz, site, …
_site/               generated static gallery (gitignored)
STRATEGIES.md        train strategies (lgbm, deotte, …)
EXPERIMENTS.md       append-only CV / LB log
```

Columns (do not invent others): `id, Age, Annual_Income_USD, Daily_Commute_km, Number_of_Cars_Owned, Charging_Stations_Near_Home, Charging_Stations_Near_Work, Environmental_Concern_Level, Gender, City_Type, Current_Car_Type, Home_Charging_Possible, Subsidy_Available, Range_Anxiety_Level, Will_Buy_EV`.

## Tests

```bash
pytest
```
