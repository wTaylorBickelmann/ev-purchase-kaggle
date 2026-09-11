# exp0001 — deotte-freq-income-commute

**Parent:** exp0000 (CV 0.94210 ± 0.00075)

**Hypothesis:** Adding train+test value-count features for `Annual_Income_USD`
(14,667 distinct, 9.2 % at exactly 30000) and `Daily_Commute_km` (833 distinct,
21.6 % at exactly 5.0) to all three Deotte XGB models lifts blend CV by >= 0.0001.
Target rate by income count-quintile is monotone (0.222 -> 0.106), which raw
income does not expose to depth-6 trees.

**Change:** new `--freq` flag on `train --strategy deotte`; `FeatureBuilder(freq=True)`
emits `Annual_Income_USD_cnt`, `Daily_Commute_km_cnt`. XGB params, folds (5, seed 42),
recipe, base_margin and 1/3 blend unchanged.
