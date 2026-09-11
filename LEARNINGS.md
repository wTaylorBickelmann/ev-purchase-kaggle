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
