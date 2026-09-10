# Implement Chris Deotte Fable 5.1 XGB starter strategy

Source of truth: public notebooks
- https://www.kaggle.com/code/cdeotte/fable-5-1-xgb-starter
- https://www.kaggle.com/code/cdeotte/fable-5-1-eda-original-data-insights
- Repo notes: `TOP20_PUBLIC_NOTEBOOKS.md`, `CURSOR.md`

## Goal

Port Deotte’s strategy into this library (not a notebook dump). Keep baseline LightGBM path working. Add a first-class Deotte/XGB path that train/predict/CLI can run.

## Recipe (exact)

```text
buy_score =
  1.2 * (Annual_Income_USD / 1e5)
  + 0.6 * Environmental_Concern_Level
  + 2.0 * (Subsidy_Available == "Yes")
  - 1.0 * (Range_Anxiety_Level == "Medium")
  - 3.0 * (Range_Anxiety_Level == "High")

buy if score + Normal(0,1) wobble > 5.5
=> P(buy) = Φ(score - 5.5)
=> base_margin = logit(clip(Φ(score-5.5), 1e-6, 1-1e-6))
```

## Helper features (exact)

```text
worry_score = Daily_Commute_km
  - 5 * Charging_Stations_Near_Home
  - 5 * Charging_Stations_Near_Work
  - 150 * (Home_Charging_Possible == "Yes")

chargers_total = Charging_Stations_Near_Home + Charging_Stations_Near_Work
income_x_subsidy = (Annual_Income_USD / 1e5) * (Subsidy_Available == "Yes")
concern_x_subsidy = Environmental_Concern_Level * (Subsidy_Available == "Yes")
```

Categoricals → category codes fit on train+test concat (same codes both sides), like Deotte.

## Three models, same 5-fold StratifiedKFold(seed=42)

1. **baseline** — features only
2. **base_margin** — same features + recipe logit as XGB `base_margin` / `base_margin_eval_set`; predict with same margins
3. **recipe_feature** — same features + `recipe_score` column

Equal-weight blend of the three OOF and test preds.

## XGB params (from notebook)

```python
dict(
  n_estimators=3000,
  learning_rate=0.05,
  max_depth=6,
  subsample=0.8,
  colsample_bytree=0.8,
  tree_method="hist",
  device="cpu",  # use cuda only if cupy/gpu available; default cpu for portability
  eval_metric="auc",
  early_stopping_rounds=100,
)
```

## Architecture constraints (`CURSOR.md`)

- Prefer small modular readable code; few LOC; type hints on public APIs.
- **Feature engineering** and **modeling** each get **classes** (not only free functions).
- Notebooks/CLI call library; no spaghetti.
- Do not commit secrets or full CSVs.
- Do not rewrite old `EXPERIMENTS.md` entries; new train runs may append.
- Keep existing LGBM baseline path intact (`python -m ev_s6e9 train` default).

## Concrete implementation plan

1. Refactor/extend `features.py` with a `FeatureBuilder` (or `DeotteFeatures`) class:
   - prep matrix with helper features
   - `recipe_score`, `recipe_logit` methods
   - category coding that can fit on combined frames
2. Extend `model.py` with an `XGBModel` / factory for Deotte params (add `xgboost` dependency in `pyproject.toml` + `requirements.txt`).
3. Add training path for Deotte 3-model blend, e.g. `train_deotte` / strategy flag:
   - 5-fold CV
   - save OOF, test preds, fold models (organized: m1/m2/m3 or blend artifacts)
   - log CV for blend (+ optionally each model) to EXPERIMENTS.md
4. Wire CLI:
   - `python -m ev_s6e9 train --strategy deotte` (name can be `deotte` / `xgb-deotte`)
   - `predict` must work for that strategy’s saved artifacts
5. Tests:
   - unit tests for recipe_score / logit / helper features on tiny frames
   - smoke train on `--synth` with tiny n_estimators override so CI is fast
6. Update README briefly for the new strategy flag.

## Verify before finishing

```bash
cd ~/Documents/code_projects/ev-purchase-kaggle
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
pytest
python -m ev_s6e9 train --synth --strategy deotte   # or whatever flag you chose
```

Synth path should complete and write outputs. If real `data/raw/train.csv` exists, also support full train (may be slow — don’t block on full data if missing).

## Out of scope

- Ravi rank-blend meta stack
- Full multi-family GBDT+TabM ensemble
- Committing trained models or large data
