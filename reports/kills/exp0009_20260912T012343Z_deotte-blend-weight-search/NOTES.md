# exp0009 — deotte-blend-weight-search

**Parent:** exp0001 (CV 0.94333 ± 0.00070, `--freq`)

**Hypothesis:** m1 (features), m2 (recipe base_margin) and m3 (recipe feature) are not
equally strong, so replacing the fixed 1/3 blend with weights chosen on OOF lifts blend
CV by >= 0.0001.

**Change:** new `--blend-search` flag on `train --strategy deotte`. Weights are searched
on a simplex grid (step 0.05, 231 combos) maximizing OOF AUC. CV is kept honest: the
weights used to score fold k are chosen on the other 4 folds' OOF (nested LOFO). The
full-OOF argmax weights are saved to `outputs/blend_weights.json` and used at predict
time. Features (`--freq`), XGB params, recipe, base_margin, folds (5, seed 42) unchanged.
