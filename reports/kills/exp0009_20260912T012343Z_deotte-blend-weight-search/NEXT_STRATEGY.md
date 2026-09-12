# exp0009 — deotte-blend-weight-search (parent exp0001, CV 0.94333 ± 0.00070)

**Hypothesis:** m1/m2/m3 are not equally strong; replacing the fixed 1/3 blend
with weights chosen on OOF (nested, leave-one-fold-out) lifts CV by >= 0.0001.

**Why now:** STRATEGY queue item 2, never tried. Tiny code change (2 files),
no new model training cost, no new features. Item 5 (TE) timed out twice —
keep this one small.

## Exact one change
Add `--blend-search` flag. When set, `run_cv` picks blend weights on a simplex
grid (step 0.05) maximizing OOF AUC. Everything else (features, `--freq`,
XGB params, folds=5, seed=42, recipe, base_margin) is unchanged.

## Files to touch (exactly 2)
- `src/ev_s6e9/deotte.py`
- `src/ev_s6e9/__main__.py`

## Steps
1. `deotte.py` — add two module-level helpers (after `_fit_fold`):
```python
def _weight_grid(step: float = 0.05) -> np.ndarray:
    n = int(round(1 / step)); g = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            g.append((i * step, j * step, 1.0 - (i + j) * step))
    return np.array(g)

def search_weights(oofs: np.ndarray, y: np.ndarray, idx: np.ndarray | None = None) -> np.ndarray:
    """oofs shape (3, n). Return weights (3,) maximizing AUC on rows idx (default all)."""
    idx = np.arange(len(y)) if idx is None else idx
    best_w, best_a = None, -1.0
    for w in _weight_grid():
        a = auc(y[idx], (w[:, None] * oofs[:, idx]).sum(axis=0))
        if a > best_a:
            best_a, best_w = a, w
    return best_w
```
2. `deotte.py` — `DeotteCvResult`: add field `weights: np.ndarray | None = None`
   (after `params`). `run_cv` and `train`: add kwarg `blend_search: bool = False`
   and pass it through (`train` → `run_cv`).
3. `deotte.py` — in `run_cv`, replace `blend_oof = np.mean(oofs, axis=0)` and the
   blend-scores loop with:
```python
    oofs_arr = np.stack(oofs)
    splits = list(skf.split(train, y))          # y and skf defined as before
    if blend_search:
        weights = search_weights(oofs_arr, y)   # full-OOF weights -> saved for test
        blend_oof = np.zeros(len(y)); blend_scores = []
        for tr, va in splits:                   # nested: weights chosen on other 4 folds
            w_k = search_weights(oofs_arr, y, idx=tr)
            blend_oof[va] = (w_k[:, None] * oofs_arr[:, va]).sum(axis=0)
            blend_scores.append(auc(y[va], blend_oof[va]))
    else:
        weights = np.full(3, 1 / 3); blend_oof = oofs_arr.mean(axis=0)
        blend_scores = [auc(y[va], blend_oof[va]) for _, va in splits]
```
   Return `DeotteCvResult(..., fb, params, weights)`.
4. `deotte.py` — `save_run`: add `"blend_weights": [float(w) for w in cv.weights]`
   to `payload`, and write `out / "blend_weights.json"` with that list.
   `train`: also `print("blend weights:", cv.weights)`.
5. `deotte.py` — `predict_proba`: load `out / "blend_weights.json"` if it exists
   (else equal weights) and return `np.average(preds, axis=0, weights=w)`.
6. `__main__.py` — add `t.add_argument("--blend-search", action="store_true",
   help="pick m1/m2/m3 blend weights on OOF (nested grid search)")` next to
   `--freq`; pass `blend_search=args.blend_search` into `train_deotte(...)`.
7. Update `exps/exp0009/config.json` and `NOTES.md` per `iteration_plan.json`.
   Smoke: `python -m ev_s6e9 train --strategy deotte --freq --blend-search --synth --no-log`.

## train_args
`["--strategy","deotte","--freq","--blend-search","--note","exp0009: OOF-searched m1/m2/m3 blend weights (nested LOFO)"]`

## Do NOT
- Change metric, fold split (5, seed 42), XGB params, features, or recipe.
- Pick weights on the fold being scored (must use `idx=tr` inside the loop).
- Touch `features.py`, `model.py`, or add a 4th model. Do not train/submit.
