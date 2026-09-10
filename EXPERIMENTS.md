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
