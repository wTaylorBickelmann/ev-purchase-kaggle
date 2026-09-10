# Loop coding agent (exp factory)

You implement **one** change per invocation inside a single new `exps/expNNNN/` folder.

## Mission

Read `outputs/RUN_BRIEF.md` (task card). Beat **current best CV** with exactly one hypothesis.

## Do

1. Read BRIEF → `STRATEGY.md` queue → `LEARNINGS.md` → `reports/index.md` → parent exp NOTES/config.
2. Pick the highest-priority **unchecked** STRATEGY item not already killed for the same idea.
3. Edit **`exps/expNNNN/config.json`** + **`NOTES.md`** (and minimal shared lib only if required).
4. One change only. Stop. Orchestrator runs `python scripts/run_exp.py expNNNN`.

## Do not

- Train full real data / Kaggle submit
- Edit other keep exps or rewrite index history
- Change metric or fold split unless STRATEGY allows
- Retry ideas already in LEARNINGS without a new angle
- Multi-unrelated rewrites

## config.json

Must include `id`, `title`, `hypothesis`, `parent`, `strategy` or `train_args`, `status: wip`.
