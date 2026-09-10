# Autonomous experiment loop (local DeepSeek)

Token-cheap overnight loop for **playground-series-s6e9**.

- **Brain:** Ollama model (default `deepseek-r1:70b`) via **Qwen Code CLI** (`qwen`)
- **Orchestrator:** `scripts/autoloop.py` (CV gate, git, optional Kaggle submit)
- **No Cursor tokens** on the hot path

## One-time setup

```bash
cd ~/Documents/code_projects/ev-purchase-kaggle
source .venv/bin/activate

# data
python -m ev_s6e9 download

# model (~42GB). M3 Ultra 512GB can go larger later; 70B is the practical coding default.
ollama pull deepseek-r1:70b

# optional: register model in Qwen settings (also done by scripts/setup_loop_model.sh)
bash scripts/setup_loop_model.sh
```

Qwen is already on PATH (`~/.local/bin/qwen`) and uses Ollama at `http://127.0.0.1:11434/v1` with `approvalMode: yolo`.

## Run

```bash
# sanity: write brief only
python scripts/autoloop.py --dry-run

# overnight (submit only on CV personal best; push commits to origin)
python scripts/autoloop.py \
  --model deepseek-r1:70b \
  --max-iters 30 \
  --min-delta 0.0001 \
  --submit \
  --push \
  --on-reject reset \
  --stop-after-no-improve 8 \
  --target-cv 0.94672

# while DeepSeek is still downloading, use Qwen 27B:
python scripts/autoloop.py --model qwen3.8:27b-q4_K_M --max-iters 5 --push
```

Work happens on branch **`loop/auto`** (created if missing). Accepts get commits like:

`loop: ACCEPT <title> CV=0.94xxx +submit`

### Flags that matter

| Flag | Meaning |
|------|---------|
| `--submit` | Kaggle submit **only** when CV ≥ best + `--min-delta` |
| `--push` | `git push -u origin loop/auto` after accepts (and keep-rejects) |
| `--on-reject reset` | `git reset --hard` to last accept (default; recommended for local models) |
| `--on-reject keep` | commit failed attempts for forensics |
| `--no-agent` | skip coder; run train from existing `outputs/next_experiment.json` |
| `--dry-run` | write `outputs/RUN_BRIEF.md` only |

State: `outputs/loop_state.json` (best CV, accept SHA, counters).

## Agent contract

Each iteration the orchestrator writes `outputs/RUN_BRIEF.md`. The coding agent must:

1. Implement **one** hypothesis
2. Write `outputs/next_experiment.json`:

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

3. Stop — orchestrator runs `python -m ev_s6e9 train …`, reads `outputs/cv.json`, gates submit

Full agent rules: `prompts/loop_agent.md` (also inlined into the brief).

## Gate policy (your rules)

1. **Optimize local CV** (`outputs/cv.json` → `mean`) from the real train path
2. **Submit only** if `new_cv >= best_cv + min_delta` (default +0.0001)
3. **Commit on progress** (accept always; reject only if `--on-reject keep`)
4. Agent may thrash the tree; **reset** restores last accept SHA
5. Never force-push `main`

## Suggested search order

1. Close gap toward Deotte public ~0.9467 (parity, seeds, blend weights)
2. Multi-seed + LGBM/XGB/CatBoost stack
3. Ideas from `TOP20_PUBLIC_NOTEBOOKS.md`
4. Only then free exploration

## Logs

- `logs/loop/agent_*.log` — coder stdout
- `logs/loop/last_train.stdout.txt`
- `outputs/RUN_BRIEF.md` — last brief
- `outputs/best_cv.json` — last accepted CV snapshot

## Swap models

```bash
ollama pull deepseek-r1:70b
# later, larger/coding-specialized tags as you like:
# ollama pull <other-tag>
python scripts/autoloop.py --model <tag> ...
```

Removing `qwen3.8:27b-q4_K_M` does **not** speed DeepSeek; only free disk if you want it back.

## Safety

- `data/raw/*` gitignored — still don’t paste keys into commits
- Daily Kaggle submit caps still apply; loop submits at most once per **accepted** CV PB
- Stop the loop with Ctrl-C; state file keeps best CV for resume
