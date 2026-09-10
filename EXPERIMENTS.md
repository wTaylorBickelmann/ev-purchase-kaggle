# Experiments

Append-only running log for playground-series-s6e9. **Newest entries at the bottom.**

Append a chunk after every attempt (or let `python -m ev_s6e9 train` / submit helpers do it). Keep it terse. Do not edit old entries.

**Rule:** once Kaggle returns a public score for that attempt, the chunk must include it under `LB:` (poll `kaggle competitions submissions` after submit if needed). Leave `LB: —` only while scoring is still pending.

Template:

```
### YYYY-MM-DD — <model / features / key params>
- CV: <mean AUC ± std>
- LB: <public score or — while pending>
- Takeaway: <one line>
```

---

### 2026-09-09 — LightGBM baseline (pipeline seed)
- Model: LightGBM, 5-fold stratified, raw cols + `charging_total`
- Params: n_estimators=800, lr=0.05, num_leaves=31, early_stopping=50
- CV: pending — run `python -m ev_s6e9 train` after `download` (appends a scored chunk below)
- LB: —
- Takeaway: first baseline scaffold; fill CV from a real-data train run
### 2026-09-09 — LightGBM n_estimators=800 lr=0.05 num_leaves=31, raw+charging_total, 5-fold
- CV: 0.94171 ± 0.00073
- LB: —
- Takeaway: auto-logged from train

### 2026-09-09 — first Kaggle submit (same LGBM baseline)
- CV: 0.94171 ± 0.00073
- LB: 0.94150
- Takeaway: CV≈LB; solid first score, room vs ~0.95 leaders

### 2026-09-09 — Chris Deotte Fable 5.1 strategy ported (`--strategy deotte`)
- Model: 3× XGB (baseline + recipe base_margin + recipe feature), equal-weight blend; helpers + EV recipe from original-data EDA
- Params: n_estimators=3000 lr=0.05 max_depth=6, 5-fold seed 42 (see STRATEGIES.md)
- CV: — (library + synth smoke only; run full `train --strategy deotte` on real data)
- LB: —
- Takeaway: Deotte path listed in STRATEGIES.md; full-data score still needed
### 2026-09-10 — Deotte XGB 3-model blend (n_estimators=3000 lr=0.05 max_depth=6), 5-fold
- CV: 0.94210 ± 0.00075
- LB: —
- Takeaway: Chris Deotte Fable 5.1 full-data train

### 2026-09-10 — Kaggle submit deotte xgb 3-model blend (Fable 5.1)
- CV: 0.94210 ± 0.00075
- LB: 0.94182
- Takeaway: CV≈LB; tiny lift vs LGBM baseline (0.94150), still well below Deotte public 0.94672

