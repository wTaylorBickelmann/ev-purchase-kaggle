# NEXT STRATEGY — exp0008 — `deotte-plus-lgbm-m4`

**Parent:** exp0001 (CV 0.94333 ± 0.00070, gate 0.94332735). **needs_code: true** — 2 files, 10 tiny edits.

## Hypothesis / why now
Add one LightGBM model (`m4`, same features as m1) as a 4th equal-weight member of the Deotte
XGB blend → CV +≥0.0001. m1/m2/m3 are all XGB on the same matrix (0.94310/0.94317/0.94309, ~0.99
correlated); a second tree library is the cheapest source of decorrelated error.
Queue #1 (3-seed) is killed; #2 (weights) is noise (variants within 0.0001); #3 = this item, never
CV-scored (3 executor timeouts on a 10-page plan). This plan was dry-run on synthetic data: passes.

## Exact ONE change
New flag `--lgbm` on `train --strategy deotte`. If set, also train `m4 = LGBMClassifier(
num_leaves=63, n_estimators=3000, lr=0.05, subsample=0.8, subsample_freq=1, colsample=0.8,
early_stopping=100)` per fold on the m1 features; blend = equal mean of all trained variants.
`predict` reads the trained-variant list from `outputs/cv.json`. Folds/seed/metric/XGB/recipe unchanged.

## Files to touch
1. `src/ev_s6e9/deotte.py` (8 edits)  2. `src/ev_s6e9/__main__.py` (2 edits)
`exps/exp0008/config.json` + `NOTES.md` are already seeded from the plan — leave as is.

## Steps — `src/ev_s6e9/deotte.py` (each FIND is unique; edit top→bottom)
1. After `import joblib` add line: `import lightgbm as lgb`
2. In the `from ev_s6e9.model import ...` line, add `make_model` → `XGB_DEFAULTS, make_model, make_xgb_model, short_xgb_params`
3. `VARIANTS = ("m1", "m2", "m3")` → `VARIANTS = ("m1", "m2", "m3", "m4")`
4. In `class DeotteVariant`, after `RECIPE_FEATURE = "m3"` add: `    LGBM = "m4"`
5. In `_fit_fold`, right BEFORE `    m = make_xgb_model(seed=seed, **overrides)` insert:
```python
    if variant == DeotteVariant.LGBM:
        m = make_model(seed=seed, **{"n_estimators": 3000, "num_leaves": 63, "subsample_freq": 1, **overrides})
        m.fit(x_tr, y_tr, eval_set=[(x_va, y_va)], callbacks=[lgb.early_stopping(100, verbose=False)])
        return m, m.predict_proba(x_va)[:, 1]
```
6. `run_cv`: add param `    lgbm: bool = False,` after `    freq: bool = False,`; and change
   `    for v in DeotteVariant:` to
```python
    for v in DeotteVariant:
        if v == DeotteVariant.LGBM and not lgbm:
            continue
```
7. `train`: add param `    lgbm: bool = False,` after `    freq: bool = False,`; in its `run_cv(...)` call
   append `, lgbm=lgbm` after `freq=freq`; and change its print loop `    for name in VARIANTS:` →
   `    for name in cv.variants:` (the one followed by `vc = cv.variants[name]`).
8. `predict_proba`: change `    for name in VARIANTS:` (followed by `paths = sorted(`) →
   `    for name in json.loads((out / CV_JSON.name).read_text(encoding="utf-8"))["variants"]:`

## Steps — `src/ev_s6e9/__main__.py`
9. After the `t.add_argument("--freq", ...)` line add:
   `    t.add_argument("--lgbm", action="store_true", help="add LightGBM m4 to the Deotte blend")`
10. In the `train_deotte(` call, after `                freq=args.freq,` add `                lgbm=args.lgbm,`

## Smoke (one command, <30 s, writes only to a temp dir)
```bash
python -c "
import tempfile;from pathlib import Path;from ev_s6e9.data import synth;from ev_s6e9 import deotte
o=Path(tempfile.mkdtemp());df=synth(800,seed=0);te=synth(200,seed=1,target=False,start_id=800)
cv=deotte.train(df,te,folds=3,log=False,out=o,model_overrides={'n_estimators':40},freq=True,lgbm=True)
assert set(cv.variants)=={'m1','m2','m3','m4'};assert deotte.predict_proba(te,out=o).shape==(200,);print('ok')"
```

## Config (already in exps/exp0008/config.json)
`train_args = ["--strategy","deotte","--freq","--lgbm","--note","exp0008: +LightGBM m4 in Deotte blend"]`,
`predict_args = ["--strategy","deotte"]`, folds 5, seed 42, parent exp0001.

## Do NOT
- Change folds/seed/metric, XGB params, recipe, base_margin, `--freq`, or blend weights (equal mean only).
- Touch `features.py`, `model.py`, `train.py`, `predict.py`, `schema.py`, `tests/`, STRATEGY/LEARNINGS.
- Run full-data `train`, `train --synth` (overwrites outputs/), `predict`, `submit`, or pytest.
- Re-plan or add extra models. Stop after the 10 edits + smoke prints `ok`.
