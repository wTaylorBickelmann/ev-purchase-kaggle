# Autonomous experiment loop — Deotte / BirdCLEF-style factory

Local **DeepSeek-V4-Flash Q3** + Qwen Code. **Files are memory** (fresh agent context each iter).

Inspired by medal loops (Lin-Chieh Huang BirdCLEF, Chris Deotte playground, etc.):
score-gated copy → one change → train → keep/kill — not “chat until gold.”

## Layout

```
STRATEGY.md          # human queue + forbidden (leak rules)
LEARNINGS.md         # append-only failures
reports/index.md     # id | idea | CV | LB | keep/kill
exps/
  exp0000/           # accepted floor (Deotte blend)
    config.json
    NOTES.md
    metrics.json
  exp0001/           # copy + one diff
scripts/
  run_exp.py         # config → train → metrics.json (prints cv_score=...)
  autoloop.py        # outer factory
```

## Cycle (orchestrator)

1. Copy last **keep** exp → `expNNNN+1`
2. Write short `outputs/RUN_BRIEF.md` (STRATEGY + LEARNINGS + index)
3. Agent edits **only** the new exp (one hypothesis)
4. `python scripts/run_exp.py expNNNN+1` → `metrics.json`
5. **Keep** iff `cv_mean >= best + ε` → commit/push; optional Kaggle submit  
   else **kill** → LEARNINGS + index row; reset tree to last accept
6. Repeat

## Run

```bash
# terminal A
bash scripts/serve_v4_flash_q3.sh   # ctx 65k

# terminal B
source .venv/bin/activate
python scripts/autoloop.py --dry-run          # copy + brief only
python scripts/autoloop.py --max-iters 20 --submit --push
```

Qwen fallback:
```bash
python scripts/autoloop.py --model qwen3.8:27b-q4_K_M \
  --base-url http://127.0.0.1:11434/v1 --max-iters 5 --push
```

## Standing rules

| Rule | |
|------|--|
| One change per iteration | yes |
| Do not edit metric / fold protocol | yes (unless STRATEGY says) |
| Failures → LEARNINGS.md | yes |
| Human owns STRATEGY + CV design | yes |
| Agent owns implement in new exp folder | yes |
| State on disk not chat | yes |

## Human knobs

- Edit **`STRATEGY.md`** queue (check off ideas; add forbidden)
- Set `"stop": "1"` in `outputs/loop_state.json` to halt
- Pause: kill autoloop; leave llama-server if you want

## Logs

- `logs/loop/autoloop_stdout.log`
- `logs/loop/agent_*.log`
- `reports/index.md`
