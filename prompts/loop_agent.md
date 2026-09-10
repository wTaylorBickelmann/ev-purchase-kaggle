# Loop coding agent

You implement **one** Kaggle experiment per invocation for `playground-series-s6e9`.

## Mission

Beat **current best local CV** (see `outputs/RUN_BRIEF.md`). Orchestrator trains, gates, and may submit.

## Do

1. Read `outputs/RUN_BRIEF.md`, `CURSOR.md`, `STRATEGIES.md`, `EXPERIMENTS.md` (tail), `TOP20_PUBLIC_NOTEBOOKS.md`.
2. Pick **one** hypothesis with a shot at real lift (features / recipe / seeds / blend / stack / notebook idea).
3. Implement cleanly behind a strategy flag or focused param change.
4. Update `STRATEGIES.md` if you add a strategy.
5. Run cheap checks: `pytest`, optional `python -m ev_s6e9 train --synth --strategy …`.
6. Write `outputs/next_experiment.json` and **stop**.

## Do not

- Run full real-data train (orchestrator does that) unless asked
- Call Kaggle submit
- Burn time on tiny hyperparam jitter when structural ideas remain
- Rewrite old `EXPERIMENTS.md` entries
- Commit secrets or competition CSVs
- Force-push / switch off the loop branch

## next_experiment.json

```json
{
  "title": "short-slug",
  "hypothesis": "one sentence",
  "strategy": "deotte",
  "train_args": ["--strategy", "deotte", "--note", "takeaway"],
  "predict_args": ["--strategy", "deotte"],
  "submit_message": "message if submitted"
}
```

## Architecture reminders

- Feature engineering and models: **classes** in `src/ev_s6e9/`
- CLI stays thin: `python -m ev_s6e9 train|predict|submit`
- Train must write `outputs/cv.json` with numeric `"mean"` (existing paths already do)
