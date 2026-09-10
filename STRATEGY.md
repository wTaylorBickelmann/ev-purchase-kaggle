# STRATEGY — human-owned experiment queue

Standing plan for the autonomous loop (playground-series-s6e9, ROC-AUC).

**Files are memory.** The agent gets a fresh context every iteration and must re-read:
`STRATEGY.md`, `LEARNINGS.md`, `reports/index.md`, and the latest accepted `exps/expXXXX/`.

## North star
- Beat local best CV, then climb toward Deotte public ~**0.94672**.
- Do **not** optimize the public LB in the inner loop. Submit only on CV personal best (orchestrator).

## Forbidden (leak / self-cheat)
- Do not change the **metric** (must stay ROC-AUC on fixed stratified 5-fold unless STRATEGY says otherwise).
- Do not change the **fold split** (seed 42, n=5 stratified on target) without an explicit STRATEGY line.
- Do not peek at test labels / use test target.
- Do not rewrite old `reports/index.md` rows or old `EXPERIMENTS.md` chunks.
- Do not “fix” a bad score by tightening early stopping only to inflate OOF without a real idea.

## One-change rule
Each iteration: **exactly one** hypothesis. Copy last accepted exp → new folder → edit the copy only (plus minimal shared lib if a new strategy flag is required).

## Queue (human edits this; agent consumes top unchecked)

Priority order — try in order, skip items already in `reports/index.md` as keep or kill:

1. [ ] Multi-seed Deotte blend (seeds 42/43/44), equal average OOF/test
2. [ ] Deotte blend weight search on OOF (m1/m2/m3 weights ≠ 1/3)
3. [ ] Add CatBoost or LightGBM as 4th model in blend with Deotte XGBs
4. [ ] Stronger recipe features from original-data EDA (interactions already partially in)
5. [ ] Target-encode high-cardinality cats with fold-safe TE
6. [ ] Hill-climb stack: LGBM OOF + Deotte OOF logistic/ridge meta on OOF only
7. [ ] Mine `TOP20_PUBLIC_NOTEBOOKS.md` for one concrete unused idea

## Stop lines (orchestrator / human)
- `STOP=1` in `outputs/loop_state.json` or max iters / no-improve streak.
- Daily Kaggle submit budget exhausted.
- Human sets **pause** — do not invent stop reasons to quit early.

## Agent role vs human role
| Human | Agent |
|------|--------|
| This file, CV protocol, leakage | Implement one change in new `exps/expNNNN/` |
| When to stop stacking junk | Train via orchestrator, log metrics |
| Which public notebook is real baseline | Append LEARNINGS on failure |

## Current accepted floor
See `reports/index.md` (exp0000 Deotte blend CV **0.94210**).
