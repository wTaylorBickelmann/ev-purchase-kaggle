# Strategies (S6E9)

Train/predict strategies exposed by `python -m ev_s6e9`.

| Flag | Name | What it does | Status |
|------|------|--------------|--------|
| `--strategy lgbm` (default) | LightGBM baseline | 5-fold stratified LGBM on raw cols + `charging_total` | Implemented — CV 0.94171 ± 0.00073, LB 0.94150 |
| `--strategy deotte` | Chris Deotte Fable 5.1 XGB blend | 3× XGBoost (baseline / recipe `base_margin` / recipe feature), equal-weight blend | Implemented — CV 0.94210 ± 0.00075, LB 0.94182 |

## Chris Deotte (`deotte`)

Source:

- [Fable 5.1 — XGB Starter](https://www.kaggle.com/code/cdeotte/fable-5-1-xgb-starter)
- [Fable 5.1 — EDA / original-data insights](https://www.kaggle.com/code/cdeotte/fable-5-1-eda-original-data-insights)
- Research notes: `TOP20_PUBLIC_NOTEBOOKS.md` (rank 1, public LB 0.94672)

### Recipe

```text
buy_score =
  1.2 * (Annual_Income_USD / 1e5)
  + 0.6 * Environmental_Concern_Level
  + 2.0 * (Subsidy_Available == Yes)
  - 1.0 * (Range_Anxiety_Level == Medium)
  - 3.0 * (Range_Anxiety_Level == High)

P(buy) ≈ Φ(score − 5.5)
base_margin = logit(clip(P, 1e-6, 1−1e-6))
```

### Helper features

- `worry_score` = commute − 5×chargers_home − 5×chargers_work − 150×home_charging
- `chargers_total`
- `income_x_subsidy`
- `concern_x_subsidy`

### Three models (same 5-fold, seed 42)

1. **m1** — features only  
2. **m2** — recipe as XGB **base_margin**  
3. **m3** — `recipe_score` as extra column  

Equal-weight blend of OOF / test preds.

### Code

- `src/ev_s6e9/features.py` — `FeatureBuilder`
- `src/ev_s6e9/deotte.py` — CV train + predict blend
- `src/ev_s6e9/model.py` — XGB defaults (n_estimators=3000, lr=0.05, max_depth=6, …)

### Run

```bash
python -m ev_s6e9 download
python -m ev_s6e9 train --strategy deotte
python -m ev_s6e9 predict --strategy deotte
python -m ev_s6e9 submit -m "deotte xgb 3-model blend"
```

Scores land in append-only `EXPERIMENTS.md` after a real (non-`--synth`) train / scored submit.

## LightGBM baseline (`lgbm`)

```bash
python -m ev_s6e9 train
python -m ev_s6e9 predict
python -m ev_s6e9 submit -m "lgbm 5-fold baseline"
```
