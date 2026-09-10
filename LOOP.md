# Autonomous experiment loop (local DeepSeek-V4-Flash Q3)

Token-cheap overnight loop for **playground-series-s6e9**.

- **Brain:** `DeepSeek-V4-Flash-0731` **UD-Q3_K_M** (~128 GB) via **llama-server**
- **Harness:** Qwen Code CLI (`qwen`) → OpenAI API at `http://127.0.0.1:8080/v1`
- **Orchestrator:** `scripts/autoloop.py` (CV gate, git, optional Kaggle submit)
- **No Cursor tokens** on the hot path

> We dropped `deepseek-r1:70b` (Ollama) in favor of V4-Flash Q3.

## One-time setup

```bash
cd ~/Documents/code_projects/ev-purchase-kaggle
source .venv/bin/activate

# data
python -m ev_s6e9 download

# ~128GB Unsloth Q3 GGUF (also: bash scripts/download_v4_flash_q3.sh)
hf download unsloth/DeepSeek-V4-Flash-0731-GGUF \
  --include 'UD-Q3_K_M/*' \
  --local-dir ~/Models/deepseek-v4-flash-q3

# llama.cpp (Metal) + Qwen provider entry
brew install llama.cpp   # already has deepseek4 arch in 0.4.0+
bash scripts/setup_loop_model.sh deepseek-v4-flash-q3 http://127.0.0.1:8080/v1
```

## Run (two terminals)

```bash
# A — serve model (alias deepseek-v4-flash-q3)
bash scripts/serve_v4_flash_q3.sh

# B — overnight loop
python scripts/autoloop.py \
  --model deepseek-v4-flash-q3 \
  --base-url http://127.0.0.1:8080/v1 \
  --max-iters 30 \
  --min-delta 0.0001 \
  --submit \
  --push \
  --on-reject reset \
  --stop-after-no-improve 8 \
  --target-cv 0.94672
```

**While Q3 is still downloading**, use local Qwen 27B:

```bash
python scripts/autoloop.py \
  --model qwen3.8:27b-q4_K_M \
  --base-url http://127.0.0.1:11434/v1 \
  --max-iters 5 --push
```

Work happens on branch **`loop/auto`**. Accepts get commits like:

`loop: ACCEPT <title> CV=0.94xxx +submit`

### Flags that matter

| Flag | Meaning |
|------|---------|
| `--submit` | Kaggle submit **only** when CV ≥ best + `--min-delta` |
| `--push` | `git push -u origin loop/auto` after accepts |
| `--on-reject reset` | `git reset --hard` to last accept (default) |
| `--on-reject keep` | commit failed attempts for forensics |
| `--no-agent` | skip coder; train from existing `outputs/next_experiment.json` |
| `--dry-run` | write `outputs/RUN_BRIEF.md` only |

State: `outputs/loop_state.json`.

## Agent contract

Orchestrator writes `outputs/RUN_BRIEF.md`. Coding agent must implement **one** hypothesis and write:

```json
{
  "title": "seed-avg-deotte",
  "hypothesis": "Average 3 seeds of deotte blend",
  "strategy": "deotte",
  "train_args": ["--strategy", "deotte", "--note", "seed avg attempt"],
  "predict_args": ["--strategy", "deotte"],
  "submit_message": "deotte multi-seed"
}
```

Rules: `prompts/loop_agent.md`, `CURSOR.md`, `QWEN.md`.

## Gate policy

1. Optimize local CV (`outputs/cv.json` → `mean`)
2. Submit only if `new_cv >= best_cv + min_delta`
3. Commit on accept; push if `--push`
4. Reject → reset to last accept SHA (default)
5. Never force-push `main`

## Paths

| What | Where |
|------|--------|
| GGUF | `~/Models/deepseek-v4-flash-q3/UD-Q3_K_M/` |
| Server | `scripts/serve_v4_flash_q3.sh` → `:8080` |
| Download | `scripts/download_v4_flash_q3.sh` |
| Agent logs | `logs/loop/agent_*.log` |

## Safety

- `data/raw/*` gitignored
- Daily Kaggle submit caps still apply (at most one submit per **accepted** CV PB)
- Stop with Ctrl-C; state file keeps best CV for resume
