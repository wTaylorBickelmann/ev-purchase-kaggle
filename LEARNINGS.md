# LEARNINGS — do not repeat

Append-only. One short bullet per failed or wasted idea. Agent and human both write here.

Template:
```
- YYYY-MM-DD expNNNN: <idea> → CV <score> vs best <best> — <why kill / what not to retry>
```

---

- (seed) 2026-09-10: bare hyperparam jitter without structural change is low-EV on this AUC plateau
- 2026-09-10 exp0001: deotte-baseline → CV 0.94210 ± 0.00075 vs best 0.94210 ± 0.00075 — kill (no real change)
- 2026-09-10: agent without `-y` under `--safe-mode` cannot write files (approval blocked) — fixed in autoloop
- 2026-09-10: cloning baseline with identical train_args is wasted — must change config
- 2026-09-11 exp0001: deotte-3seed-blend → CV 0.94212 ± 0.00075 vs best 0.94210 ± 0.00075 — kill
- 2026-09-11 exp0001: EXECUTOR FAILED — executor timed out after 3600s
- 2026-09-11 exp0002: EXECUTOR FAILED — executor timed out after 3600s
- 2026-09-11 exp0002: deotte-freq2-age-nmode → CV 0.94328 ± 0.00073 vs best 0.94333 ± 0.00070 — kill
- 2026-09-11 exp0002: EXECUTOR FAILED — executor timed out after 3600s
- 2026-09-12 exp0008: deotte-plus-lgbm-m4 → CV 0.94336 ± 0.00070 vs best 0.94333 ± 0.00070 — kill
- 2026-09-12 exp0009: deotte-blend-weight-search → CV 0.94333 ± 0.00070 vs best 0.94333 ± 0.00070 — kill
- 2026-09-12 exp0011: deotte-te-commute → CV 0.94557 ± 0.00061 vs best 0.94552 ± 0.00064 — kill
- 2026-09-12 exp0012: deotte-te-m2 → CV 0.94559 ± 0.00065 vs best 0.94552 ± 0.00064 — kill
- 2026-09-12 exp0013: deotte-lr02 → CV 0.94555 ± 0.00064 vs best 0.94552 ± 0.00064 — kill
- 2026-09-12 exp0014: PLANNER FAILED — Cursor planner timed out after 900s
- 2026-09-12 exp0015: deotte-te-pair → CV 0.94551 ± 0.00063 vs best 0.94552 ± 0.00064 — kill
- 2026-09-12 exp0016: deotte-orig-te → CV 0.94553 ± 0.00063 vs best 0.94552 ± 0.00064 — kill
- 2026-09-12 exp0017: PLANNER FAILED — Cursor planner timed out after 900s
- 2026-09-12 exp0018: PLANNER FAILED — Cursor planner exited 1; see /Users/will/Documents/code_projects/ev-purchase-kaggle/logs/loop/planner_cursor_1789193067.log
