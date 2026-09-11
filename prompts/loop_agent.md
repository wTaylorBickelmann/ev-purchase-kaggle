# Loop agents

## Planner
API model (Fable or Grok). Writes `outputs/iteration_plan.json`. No tools.

## Executor (Qwen Code)
Fresh session every iteration. Read **only**:
- `outputs/ITERATION_PLAN.md`
- `outputs/iteration_plan.json`
- `exps/expNNNN/config.json` + `NOTES.md`

Implement the plan. Minimal `src/` edits only if `needs_code`. Do not train/submit. Stop when done.
