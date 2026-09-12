# exp0010 — deotte-te-income

**Parent:** exp0001 (CV 0.94333 ± 0.00070)

**Hypothesis:** Fold-safe smoothed target encoding (m=20) of the exact `Annual_Income_USD`
value, added to all three Deotte XGBs on top of `--freq`, lifts blend CV by >= 0.0001.
Income spike values (count 600-1000 each; 68 % of rows have count >= 100) carry
value-specific target rates that are not smooth in income (68480 -> 0.270, 65800 -> 0.065,
72441 -> 0.036, 131789 -> 0.353). `--freq` only flags a spike; 256-bin hist XGB cannot
isolate 14.6k exact values.

**Change:** new `--te` flag -> column `Annual_Income_USD_te`. Train rows: nested 5-fold
OOF TE inside the outer train fold; val/test rows: map fit on the whole outer train fold
(stored per fold model as `te_map_`). XGB params, folds (5, seed 42), recipe,
base_margin and 1/3 blend unchanged.
