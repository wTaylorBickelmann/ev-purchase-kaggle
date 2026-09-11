# Autonomous experiment loop — planner + executor

Two-model factory for **playground-series-s6e9**.

| Role | Model | Job |
|------|--------|-----|
| **Planner** | **Claude Fable 5.1** via OpenRouter when `OPENROUTER_API_KEY` is set; else **Grok** via xAI | Read STRATEGY / LEARNINGS / index / parent → write one-iteration plan |
| **Executor** | **Qwen 3.8 27B** (Ollama `qwen3.8:27b-q4_K_M`) via Qwen Code CLI | **Fresh session every iteration** — implement the plan only (short context) |
| **Orchestrator** | `scripts/autoloop.py` | Copy exp → plan → (optional code) → train → CV gate keep/kill |

DeepSeek V4 is **not** on the hot path anymore (too slow / context thrash for this harness).

## Cycle

1. Copy last **keep** `exps/expNNNN` → `expNNNN+1`
2. **Planner** → `outputs/iteration_plan.json` + `ITERATION_PLAN.md`
3. Orchestrator writes plan `config.json` + `NOTES.md`
4. If `needs_code` → **Qwen** fresh session (`-y`, no chat resume)
5. `python scripts/run_exp.py expNNNN+1` → `metrics.json`
6. Keep iff `cv_mean >= best + ε` (optional Kaggle submit); else kill + LEARNINGS

## Layout

```
STRATEGY.md          # human queue + leak rules
LEARNINGS.md         # append-only failures
reports/index.md     # keep/kill table
exps/exp0000/        # accepted floor
outputs/iteration_plan.json
outputs/ITERATION_PLAN.md
scripts/autoloop.py
scripts/run_exp.py
```

## Setup

```bash
# executor
ollama pull qwen3.8:27b-q4_K_M

# planner: either
#   export OPENROUTER_API_KEY=...   # enables anthropic/claude-fable-5.1
# or use existing XAI_API_KEY (Grok fallback) from ~/.hermes/.env
```

## Run

```bash
source .venv/bin/activate
python scripts/autoloop.py --dry-run          # planner only
python scripts/autoloop.py --max-iters 20 --submit --push
```

Override models:

```bash
# force Fable (requires OpenRouter)
EV_LOOP_PLAN_MODEL=anthropic/claude-fable-5.1 OPENROUTER_API_KEY=... \
  python scripts/autoloop.py --max-iters 5

# force Grok planner
EV_LOOP_PLAN_MODEL=grok-4.5 python scripts/autoloop.py --max-iters 5

# executor
python scripts/autoloop.py --exec-model qwen3.8:27b-q4_K_M \
  --exec-base-url http://127.0.0.1:11434/v1
```

Flags: `--no-executor` (config-only plans), `--force-executor`, `--on-reject reset|keep`.

## Why fresh Qwen sessions

Qwen degrades on long coding threads. Each executor call:
- deletes `~/.qwen/projects/.../chats` for this repo
- starts without `--continue`
- gets only the iteration plan + exp paths (not full history)

## Human knobs

- Edit **`STRATEGY.md`** queue
- `"stop": "1"` in `outputs/loop_state.json`
- Kill: `pkill -f scripts/autoloop.py`
