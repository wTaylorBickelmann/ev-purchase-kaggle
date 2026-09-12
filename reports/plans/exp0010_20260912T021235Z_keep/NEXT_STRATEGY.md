# exp0010 — deotte-te-income (parent exp0001, CV 0.94333 ± 0.00070)

**Hypothesis:** fold-safe smoothed target encoding of the exact `Annual_Income_USD`
value (added to all 3 Deotte XGBs, on top of `--freq`) lifts blend CV by >= 0.0001.

**Why now:** queue items 1–3 are killed (multi-seed, weight search, LGBM m4 all
≈ +0.00000–0.00003). Item 5 (TE) was never scored — it timed out twice on oversized
plans. EDA (read-only, this iter): 68 % of rows have an income value with count >= 100,
and those spike values carry *value-specific* target rates that are not smooth in
income — 68480 → 0.270, 65800 → 0.065, 72441 → 0.036, 131789 → 0.353. `--freq`
only says "is a spike"; hist XGB (256 bins) cannot isolate 14.6k exact values. TE
hands the per-value rate straight to the trees. Income only (commute spikes are
flat ≈ 0.14–0.19, skip).

## Exact one change
New `--te` flag → extra column `Annual_Income_USD_te`. Train rows: nested 5-fold
OOF TE inside the outer train fold. Val/test rows: map fit on the whole outer train
fold. Smoothing m=20 toward the fold prior. Nothing else changes.

## Files to touch (3, tiny edits each)
`src/ev_s6e9/features.py`, `src/ev_s6e9/deotte.py`, `src/ev_s6e9/__main__.py`

## Steps
1. `features.py` — add `from sklearn.model_selection import StratifiedKFold`, the
   constant `TE_COL = "Annual_Income_USD"`, and three module-level helpers:
```python
def te_fit(vals: pd.Series, y: np.ndarray, m: float = 20.0) -> tuple[pd.Series, float]:
    prior = float(np.mean(y))
    g = pd.DataFrame({"v": vals.to_numpy(), "y": y}).groupby("v")["y"].agg(["sum", "count"])
    return (g["sum"] + prior * m) / (g["count"] + m), prior

def te_apply(vals: pd.Series, mapping: pd.Series, prior: float) -> np.ndarray:
    return vals.map(mapping).fillna(prior).to_numpy(dtype=np.float32)

def te_oof(vals: pd.Series, y: np.ndarray, *, folds: int = 5, seed: int = 42, m: float = 20.0) -> np.ndarray:
    """Nested out-of-fold TE for training rows (no row sees its own label)."""
    out = np.zeros(len(vals), dtype=np.float32)
    for tr, va in StratifiedKFold(folds, shuffle=True, random_state=seed).split(vals, y):
        mp, prior = te_fit(vals.iloc[tr], y[tr], m)
        out[va] = te_apply(vals.iloc[va], mp, prior)
    return out
```
   `FeatureBuilder.__init__(self, freq: bool = False, te: bool = False)` → `self.te = te`.
   Do NOT add the TE column inside `transform` (it is fold-dependent).
2. `deotte.py` — import `TE_COL, te_apply, te_fit, te_oof` from `ev_s6e9.features`.
   In `_fit_fold`, right after `m = make_xgb_model(...)`:
```python
    if fb.te:
        v_tr = pd.to_numeric(raw_tr[TE_COL], errors="coerce")
        v_va = pd.to_numeric(raw_va[TE_COL], errors="coerce")
        mp, prior = te_fit(v_tr, y_tr)
        x_tr[TE_COL + "_te"] = te_oof(v_tr, y_tr, seed=seed)
        x_va[TE_COL + "_te"] = te_apply(v_va, mp, prior)
        m.te_map_ = (mp, prior)          # pickled with the fold model by joblib
```
3. `deotte.py` — `_predict_variant`: build `x` per model (move `fb.transform` inside
   the loop); if `fb.te`: `x[TE_COL + "_te"] = te_apply(pd.to_numeric(df[TE_COL], errors="coerce"), *m.te_map_)`
   before `predict_proba`. Margin branch unchanged.
4. `deotte.py` — add kwarg `te: bool = False` to `run_cv` and `train`; `train` passes
   `te=te` to `run_cv`; `run_cv` does `FeatureBuilder(freq=freq, te=te).fit(train, test)`.
5. `__main__.py` — `t.add_argument("--te", action="store_true", help="fold-safe target encoding of Annual_Income_USD")`
   next to `--freq`; pass `te=args.te` into `train_deotte(...)`.
6. Write `exps/exp0010/config.json` + `NOTES.md` per `iteration_plan.json`.
   Smoke: `python -m ev_s6e9 train --strategy deotte --freq --te --synth --no-log`
   then `python -m ev_s6e9 predict --strategy deotte` must run. Stop after edits.

## train_args
`["--strategy","deotte","--freq","--te","--note","exp0010: +fold-safe smoothed TE (m=20, nested OOF) of exact Annual_Income_USD on top of --freq"]`

## Do NOT
- Change metric, outer fold split (5, seed 42), XGB params, recipe, base_margin, 1/3 blend.
- Fit the TE map on `raw_va` or on full train inside CV (leak). Train rows must use `te_oof`.
- TE any other column, touch `model.py`, add models, or train/submit on real data.
