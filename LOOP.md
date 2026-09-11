# Autonomous experiment loop — Fable (Cursor) plans, Qwen executes

| Role | Model | Job |
|------|--------|-----|
| **Planner** | **Claude Fable 5.1** via **Cursor Agent** (`~/.local/bin/agent`) | End of each setup / start of iter: assess STRATEGY + results → write `outputs/NEXT_STRATEGY.md` + `iteration_plan.json` |
| **Executor** | **Qwen 27B** (Ollama) + Qwen Code | **Fresh session every iter** — implement only that MD/plan |
| **Orchestrator** | `scripts/autoloop.py` | Copy exp → Fable plan → Qwen code → train → CV gate |

API Grok/OpenRouter remains `--plan-backend api` fallback only.

## Cycle

1. Copy last keep → `exps/expNNNN+1`
2. **Cursor Fable** writes `outputs/NEXT_STRATEGY.md` (+ JSON)
3. Orchestrator seeds `config.json` / `NOTES.md` from the plan
4. If `needs_code` → **Qwen** fresh session implements the MD
5. `python scripts/run_exp.py expNNNN` → CV keep/kill

## Run

```bash
# dry-run planner only (Fable via Cursor)
python scripts/autoloop.py --dry-run --plan-backend cursor \
  --plan-model claude-fable-5-1-thinking-high

# full loop
python scripts/autoloop.py --max-iters 20 --submit --push
# or
bash logs/loop/start_autoloop.sh
```

## Why this split

- Fable is strong at “what next” given STRATEGY/LEARNINGS/index.
- Qwen is weak on long coding threads → wipe chats each iter; feed only `NEXT_STRATEGY.md`.
- Cursor already has Fable on your account (no OpenRouter key required).

## Durable logging (no overwrite)

| Path | What |
|------|------|
| `reports/plans/{exp}_{ts}_{title}/` | Full plan snapshot every iter |
| `reports/plans/INDEX.md` | Catalog of plans |
| `reports/kills/{exp}_{ts}_{title}/` | Killed exp config + metrics + plan |
| `reports/kills/INDEX.md` | Catalog of kills |
| `LEARNINGS.md` / `reports/index.md` / `EXPERIMENTS.md` | Restored after kill reset |

Live `outputs/NEXT_STRATEGY.md` is only the **current** pointer for Qwen; history is under `reports/plans/`.
