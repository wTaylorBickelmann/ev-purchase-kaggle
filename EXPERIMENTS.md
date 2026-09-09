# Experiments

Append-only running log for playground-series-s6e9. **Newest entries at the bottom.**

Append a chunk after every attempt (or let `python -m ev_s6e9 train` do it). Keep it terse. Do not edit old entries.

Template:

```
### YYYY-MM-DD — <model / features / key params>
- CV: <mean AUC ± std>
- LB: <score or —>
- Takeaway: <one line>
```

---

### 2026-09-09 — LightGBM baseline (pipeline seed)
- Model: LightGBM, 5-fold stratified, raw cols + `charging_total`
- Params: n_estimators=800, lr=0.05, num_leaves=31, early_stopping=50
- CV: pending — run `python -m ev_s6e9 train` after `download` (appends a scored chunk below)
- LB: —
- Takeaway: first baseline scaffold; fill CV from a real-data train run
