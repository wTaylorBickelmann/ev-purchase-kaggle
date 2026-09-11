#!/usr/bin/env python3
"""Two-model score-gated experiment factory.

Planner (strong reasoning API — Claude Fable 5.1 via OpenRouter when keyed,
else Grok via xAI): reads STRATEGY / LEARNINGS / index / parent → writes a
tight iteration plan.

Executor (local Qwen ~27–28B via Ollama + Qwen Code CLI): **fresh session every
iteration**, implements only that plan (short context).

Then orchestrator trains + CV-gates keep/kill.

Cycle:
  1. Copy last keep exp → expNNNN+1
  2. Planner → outputs/iteration_plan.json + ITERATION_PLAN.md
  3. Apply config from plan; if needs_code → Qwen fresh session
  4. python scripts/run_exp.py expNNNN+1
  5. Keep iff cv_mean >= best + ε else kill

Examples:
  python scripts/autoloop.py --dry-run
  python scripts/autoloop.py --max-iters 20 --submit --push
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
EXPS = ROOT / "exps"
REPORTS = ROOT / "reports"
INDEX = REPORTS / "index.md"
STRATEGY = ROOT / "STRATEGY.md"
LEARNINGS = ROOT / "LEARNINGS.md"
STATE_PATH = OUT / "loop_state.json"
PLAN_JSON = OUT / "iteration_plan.json"
PLAN_MD = OUT / "ITERATION_PLAN.md"
BRIEF_PATH = OUT / "RUN_BRIEF.md"  # kept for compatibility (points at plan)
LOG_DIR = ROOT / "logs" / "loop"
VENV_PY = ROOT / ".venv" / "bin" / "python"

# Executor defaults: local Qwen 27B (user: ~28B class)
DEFAULT_EXEC_MODEL = os.environ.get("EV_LOOP_EXEC_MODEL", "qwen3.8:27b-q4_K_M")
DEFAULT_EXEC_BASE = os.environ.get("EV_LOOP_EXEC_BASE_URL", "http://127.0.0.1:11434/v1")
DEFAULT_AGENT = os.environ.get("EV_LOOP_AGENT", "qwen")

# Planner: Cursor Agent + Claude Fable (primary). API Grok/OpenRouter = fallback.
DEFAULT_PLAN_BACKEND = os.environ.get("EV_LOOP_PLAN_BACKEND", "cursor")  # cursor|api
DEFAULT_PLAN_MODEL = os.environ.get(
    "EV_LOOP_PLAN_MODEL", "claude-fable-5-1-thinking-high"
)
DEFAULT_CURSOR_AGENT = os.environ.get("EV_LOOP_CURSOR_AGENT", str(Path.home() / ".local" / "bin" / "agent"))
DEFAULT_BEST = 0.94210
BRANCH = "loop/auto"
NEXT_STRATEGY = OUT / "NEXT_STRATEGY.md"  # Fable writes this each iter for Qwen

HERMES_ENV = Path.home() / ".hermes" / ".env"
QWEN_PROJECT = (
    Path.home()
    / ".qwen"
    / "projects"
    / "-Users-will-Documents-code-projects-ev-purchase-kaggle"
)

KEEP_ON_CLEAN = {
    "data",
    ".venv",
    "outputs",
    "logs",
    ".git",
    ".pytest_cache",
    "__pycache__",
    "exps",
    "reports",
}


@dataclass
class State:
    best_cv: float
    best_cv_str: str
    accept_sha: str
    best_exp: str
    iteration: int
    accepts: int
    rejects: int
    submits: int
    no_improve_streak: int
    branch: str
    updated_at: str
    stop: str = "0"

    @classmethod
    def load(cls, path: Path) -> State:
        if path.exists():
            d = json.loads(path.read_text(encoding="utf-8"))
            d.setdefault("best_exp", "exp0000")
            d.setdefault("stop", "0")
            allowed = set(cls.__dataclass_fields__.keys())
            return cls(**{k: v for k, v in d.items() if k in allowed})
        return cls(
            best_cv=0.0,
            best_cv_str="",
            accept_sha="",
            best_exp="exp0000",
            iteration=0,
            accepts=0,
            rejects=0,
            submits=0,
            no_improve_streak=0,
            branch=BRANCH,
            updated_at="",
            stop="0",
        )

    def save(self, path: Path) -> None:
        self.updated_at = datetime.now(timezone.utc).isoformat()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def py() -> str:
    return str(VENV_PY) if VENV_PY.exists() else sys.executable


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess[str]:
    print(f"$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True, **kw)


def git(*args: str, check: bool = True) -> str:
    r = run(["git", *args])
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed:\n{r.stderr or r.stdout}")
    return (r.stdout or "").strip()


def ensure_branch(branch: str) -> None:
    cur = git("rev-parse", "--abbrev-ref", "HEAD")
    if cur == branch:
        return
    if git("branch", "--list", branch):
        git("checkout", branch)
    else:
        git("checkout", "-b", branch)


def head_sha() -> str:
    return git("rev-parse", "HEAD")


def list_exps() -> list[Path]:
    if not EXPS.exists():
        return []
    exps = [p for p in EXPS.iterdir() if p.is_dir() and re.fullmatch(r"exp\d{4}", p.name)]
    return sorted(exps, key=lambda p: p.name)


def next_exp_id() -> str:
    exps = list_exps()
    if not exps:
        return "exp0000"
    n = max(int(p.name.replace("exp", "")) for p in exps) + 1
    return f"exp{n:04d}"


def load_metrics(exp_dir: Path) -> dict | None:
    m = exp_dir / "metrics.json"
    if not m.exists():
        return None
    return json.loads(m.read_text(encoding="utf-8"))


def best_from_exps() -> tuple[float, str, str]:
    best, best_s, best_exp = 0.0, "", "exp0000"
    for p in list_exps():
        met = load_metrics(p) or {}
        cfg: dict = {}
        if (p / "config.json").exists():
            cfg = json.loads((p / "config.json").read_text(encoding="utf-8"))
        if str(met.get("status") or cfg.get("status") or "").lower() == "kill":
            continue
        mean = met.get("cv_mean", cfg.get("cv_mean"))
        if mean is None:
            continue
        mean = float(mean)
        if mean > best:
            best = mean
            best_s = str(met.get("cv") or cfg.get("cv") or f"{mean:.5f}")
            best_exp = p.name
    if best <= 0:
        return DEFAULT_BEST, f"{DEFAULT_BEST:.5f}", "exp0000"
    return best, best_s, best_exp


def copy_exp(src_id: str, dst_id: str) -> Path:
    src, dst = EXPS / src_id, EXPS / dst_id
    if not src.is_dir():
        raise FileNotFoundError(src)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("models", "__pycache__", "*.joblib"))
    for name in ("metrics.json", "cv.json", "oof.csv", "submission.csv"):
        p = dst / name
        if p.exists():
            p.unlink()
    cfg_path = dst / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    cfg.update(
        {
            "id": dst_id,
            "parent": src_id,
            "status": "wip",
            "title": cfg.get("title") or dst_id,
            "hypothesis": "TBD — set by planner",
        }
    )
    for k in ("cv_mean", "cv_std", "cv", "lb"):
        cfg.pop(k, None)
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    (dst / "NOTES.md").write_text(
        f"# {dst_id}\n\n**Parent:** {src_id}\n\n**Hypothesis:** (planner/executor)\n",
        encoding="utf-8",
    )
    return dst


def _tail(path: Path, n: int) -> str:
    if not path.exists():
        return "(missing)"
    lines = path.read_text(encoding="utf-8").splitlines()
    return "\n".join(lines if len(lines) <= n else lines[-n:])


def resolve_api_planner() -> tuple[str, str, str]:
    """API fallback: (base_url, api_key, model_id). Prefer OpenRouter Fable, else Grok."""
    load_dotenv(HERMES_ENV)
    override = os.environ.get("EV_LOOP_API_PLAN_MODEL", "").strip()
    or_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    xai_key = os.environ.get("XAI_API_KEY", "").strip()
    if override:
        if override.startswith("anthropic/") or "fable" in override.lower():
            if not or_key:
                raise RuntimeError("API Fable needs OPENROUTER_API_KEY")
            return "https://openrouter.ai/api/v1", or_key, override
        if not xai_key:
            raise RuntimeError("API planner needs XAI_API_KEY")
        return "https://api.x.ai/v1", xai_key, override.replace("xai/", "")
    if or_key:
        return "https://openrouter.ai/api/v1", or_key, "anthropic/claude-fable-5.1"
    if xai_key:
        return "https://api.x.ai/v1", xai_key, "grok-4.5"
    raise RuntimeError("No API planner credentials")


def openai_chat(
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict],
    *,
    max_tokens: int = 4096,
    temperature: float = 0.3,
    timeout: int = 300,
) -> str:
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/wTaylorBickelmann/ev-purchase-kaggle",
            "X-Title": "ev-s6e9-autoloop",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"planner HTTP {e.code}: {err[:800]}") from e
    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError(f"planner empty response: {payload!r}"[:500])
    msg = choices[0].get("message") or {}
    content = msg.get("content") or ""
    if not content and msg.get("reasoning_content"):
        content = msg["reasoning_content"]
    return content.strip()


def extract_json_object(text: str) -> dict:
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        return json.loads(fence.group(1))
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError(f"no JSON object in planner output:\n{text[:1000]}")


def plan_from_files(new_exp: str, parent: str, planner_model: str) -> dict:
    """Load plan written by Cursor Fable (or synthesize from NEXT_STRATEGY.md)."""
    if PLAN_JSON.exists():
        try:
            plan = json.loads(PLAN_JSON.read_text(encoding="utf-8"))
            if isinstance(plan, dict) and (plan.get("config") or plan.get("hypothesis")):
                plan["planner_model"] = plan.get("planner_model") or planner_model
                plan["exp_id"] = new_exp
                return plan
        except Exception:
            pass
    # Minimal plan from markdown only — executor must do the heavy lift
    md = ""
    if NEXT_STRATEGY.exists():
        md = NEXT_STRATEGY.read_text(encoding="utf-8")
    elif PLAN_MD.exists():
        md = PLAN_MD.read_text(encoding="utf-8")
    title = "from-next-strategy"
    m = re.search(r"^#\s+(.+)$", md, re.M)
    if m:
        title = re.sub(r"[^\w\-]+", "-", m.group(1).strip().lower())[:48].strip("-") or title
    hyp = ""
    for key in ("Hypothesis", "hypothesis", "One change", "Change"):
        mm = re.search(rf"\*\*{key}\*\*:\s*(.+)", md)
        if mm:
            hyp = mm.group(1).strip()
            break
    if not hyp:
        hyp = (md.strip().splitlines() or ["see NEXT_STRATEGY.md"])[0][:200]
    needs = "needs_code" in md.lower() or "src/ev_s6e9" in md or "implement" in md.lower()
    cfg = {
        "id": new_exp,
        "title": title,
        "hypothesis": hyp,
        "parent": parent,
        "strategy": "deotte",
        "train_args": ["--strategy", "deotte", "--note", f"{new_exp}: {hyp[:60]}"],
        "predict_args": ["--strategy", "deotte"],
        "submit_message": new_exp,
        "folds": 5,
        "seed": 42,
        "status": "wip",
    }
    # try pull train_args fence from md
    ta = re.search(r"train_args\"?\s*:\s*(\[[^\]]+\])", md)
    if ta:
        try:
            cfg["train_args"] = json.loads(ta.group(1).replace("'", '"'))
        except Exception:
            pass
    plan = {
        "title": title,
        "hypothesis": hyp,
        "one_change": hyp,
        "rationale": "from NEXT_STRATEGY.md",
        "needs_code": needs,
        "code_instructions": md[:4000],
        "files_to_edit": [f"exps/{new_exp}/config.json", f"exps/{new_exp}/NOTES.md"],
        "config": cfg,
        "notes_md": f"# {new_exp}\n\n**Hypothesis:** {hyp}\n\nSee outputs/NEXT_STRATEGY.md\n",
        "executor_checklist": [
            "Read outputs/NEXT_STRATEGY.md fully",
            "Implement exactly one change",
            "Stop when done",
        ],
        "planner_model": planner_model,
        "exp_id": new_exp,
    }
    PLAN_JSON.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    return plan


def build_cursor_planner_prompt(state: State, new_exp: str, parent: str) -> str:
    return f"""You are the PLANNER for Kaggle playground-series-s6e9 (ROC-AUC).

Assess progress and propose the **next single experiment** for `{new_exp}` (parent `{parent}`).
Best CV so far: **{state.best_cv_str}** (gate {state.best_cv:.8f}). North star ~0.94672.

## Read (do not rewrite history)
- STRATEGY.md
- LEARNINGS.md
- reports/index.md
- STRATEGIES.md
- exps/{parent}/config.json
- exps/{parent}/NOTES.md
- outputs/loop_state.json (if present)

## Write ONLY these files (overwrite)
1. `outputs/NEXT_STRATEGY.md` — human plan for the Qwen executor:
   - Title / hypothesis / why now
   - Exact one change
   - Files to touch
   - Step-by-step implementation (Qwen is ~27B — be concrete, small diffs)
   - How train_args / config should look
   - What NOT to do (metric/folds/leak)
2. `outputs/iteration_plan.json` — machine plan, this exact schema:
```json
{{
  "title": "short-slug",
  "hypothesis": "one sentence",
  "rationale": "why",
  "one_change": "single concrete change",
  "needs_code": true,
  "code_instructions": "precise steps for Qwen",
  "files_to_edit": ["exps/{new_exp}/config.json", "exps/{new_exp}/NOTES.md"],
  "config": {{
    "id": "{new_exp}",
    "title": "short-slug",
    "hypothesis": "one sentence",
    "parent": "{parent}",
    "strategy": "deotte",
    "train_args": ["--strategy", "deotte", "--note", "{new_exp}: ..."],
    "predict_args": ["--strategy", "deotte"],
    "submit_message": "{new_exp}",
    "folds": 5,
    "seed": 42,
    "status": "wip"
  }},
  "notes_md": "# {new_exp}\\n\\n**Hypothesis:** ...\\n",
  "executor_checklist": ["...", "stop after edits"]
}}
```
3. Also copy the same markdown into `outputs/ITERATION_PLAN.md` (same content as NEXT_STRATEGY or a short pointer).

## Rules
- ONE change only. Prefer top unchecked STRATEGY item if still valid.
- Skip killed LEARNINGS ideas.
- Do NOT train, submit, or edit src/ unless the JSON needs_code instructions require the executor to later.
- You are planner only — do not implement the experiment code yourself beyond writing the plan files.
- Stop when the three outputs exist and JSON is valid.
"""


def run_planner_cursor(
    state: State,
    new_exp: str,
    parent: str,
    *,
    model: str,
    agent_bin: str,
    timeout: int,
) -> dict:
    if shutil.which(agent_bin) is None and not Path(agent_bin).exists():
        raise RuntimeError(f"Cursor agent not found: {agent_bin}")
    # clear prior plan artifacts so we don't accept stale JSON
    for p in (PLAN_JSON, PLAN_MD, NEXT_STRATEGY):
        if p.exists():
            p.unlink()
    prompt = build_cursor_planner_prompt(state, new_exp, parent)
    cmd = [
        agent_bin,
        "--print",
        "--yolo",
        "--trust",
        "--workspace",
        str(ROOT),
        "--model",
        model,
        "--output-format",
        "text",
        prompt,
    ]
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"planner_cursor_{int(time.time())}.log"
    print(f"planner (Cursor Fable) model={model} log→ {log_path}", flush=True)
    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"cmd: {' '.join(cmd[:8])} ...\n\n")
        log.flush()
        p = subprocess.Popen(
            cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, text=True
        )
        try:
            rc = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            raise RuntimeError(f"Cursor planner timed out after {timeout}s")
    print(f"planner cursor done in {time.time() - t0:.1f}s rc={rc}", flush=True)
    if rc != 0 and not (PLAN_JSON.exists() or NEXT_STRATEGY.exists()):
        raise RuntimeError(f"Cursor planner exited {rc}; see {log_path}")
    if not NEXT_STRATEGY.exists() and PLAN_MD.exists():
        shutil.copy2(PLAN_MD, NEXT_STRATEGY)
    if not NEXT_STRATEGY.exists() and not PLAN_JSON.exists():
        raise RuntimeError(f"planner wrote neither NEXT_STRATEGY.md nor plan JSON; see {log_path}")
    plan = plan_from_files(new_exp, parent, model)
    # ensure NEXT_STRATEGY exists for Qwen
    if not NEXT_STRATEGY.exists() and PLAN_MD.exists():
        shutil.copy2(PLAN_MD, NEXT_STRATEGY)
    if not PLAN_MD.exists() and NEXT_STRATEGY.exists():
        shutil.copy2(NEXT_STRATEGY, PLAN_MD)
    plan["planner_backend"] = "cursor"
    PLAN_JSON.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    BRIEF_PATH.write_text(
        f"# RUN BRIEF\n\nPlanner: Cursor `{model}`\n\n"
        f"Executor: read `outputs/NEXT_STRATEGY.md` + `iteration_plan.json`.\n",
        encoding="utf-8",
    )
    return plan


def run_planner_api(state: State, new_exp: str, parent: str) -> dict:
    base, key, model = resolve_api_planner()
    # reuse a compact API prompt
    parent_cfg = ""
    pcfg = EXPS / parent / "config.json"
    if pcfg.exists():
        parent_cfg = pcfg.read_text(encoding="utf-8")
    prompt = f"""Kaggle s6e9 planner. Best CV {state.best_cv_str}. New exp {new_exp} parent {parent}.
Read mental context:
STRATEGY:\n{_tail(STRATEGY, 80)}\nLEARNINGS:\n{_tail(LEARNINGS, 30)}\nINDEX:\n{_tail(INDEX, 20)}\nPARENT:\n{parent_cfg}

Return ONLY JSON with keys title,hypothesis,rationale,one_change,needs_code,code_instructions,files_to_edit,config,notes_md,executor_checklist.
config.id={new_exp} config.parent={parent} config.status=wip.
"""
    print(f"planner (API) model={model} base={base}", flush=True)
    text = openai_chat(
        base,
        key,
        model,
        [
            {"role": "system", "content": "Output valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=4096,
        temperature=0.2,
    )
    (LOG_DIR / f"planner_api_{int(time.time())}.txt").write_text(text, encoding="utf-8")
    plan = extract_json_object(text)
    plan.setdefault("needs_code", False)
    cfg = plan.get("config") or {}
    cfg["id"] = new_exp
    cfg["parent"] = parent
    cfg["status"] = "wip"
    plan["config"] = cfg
    plan["planner_model"] = model
    plan["planner_backend"] = "api"
    plan["exp_id"] = new_exp
    PLAN_JSON.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    md = [
        f"# NEXT STRATEGY — {new_exp}",
        "",
        f"**Planner:** `{model}` (API)",
        f"**Title:** {plan.get('title')}",
        f"**Hypothesis:** {plan.get('hypothesis')}",
        f"**One change:** {plan.get('one_change')}",
        f"**needs_code:** {plan.get('needs_code')}",
        "",
        "## Rationale",
        str(plan.get("rationale") or ""),
        "",
        "## Code instructions",
        str(plan.get("code_instructions") or "(none)"),
        "",
        "## config",
        "```json",
        json.dumps(cfg, indent=2),
        "```",
        "",
    ]
    NEXT_STRATEGY.write_text("\n".join(md) + "\n", encoding="utf-8")
    PLAN_MD.write_text(NEXT_STRATEGY.read_text(encoding="utf-8"), encoding="utf-8")
    return plan


def run_planner(
    state: State,
    new_exp: str,
    parent: str,
    *,
    backend: str,
    model: str,
    cursor_bin: str,
    timeout: int,
) -> dict:
    if backend == "cursor":
        return run_planner_cursor(
            state, new_exp, parent, model=model, agent_bin=cursor_bin, timeout=timeout
        )
    if backend == "api":
        return run_planner_api(state, new_exp, parent)
    raise ValueError(f"unknown plan backend: {backend}")


def apply_plan_files(plan: dict, new_exp: str) -> None:
    exp_dir = EXPS / new_exp
    cfg = plan.get("config") or {}
    cfg["id"] = new_exp
    (exp_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    notes = plan.get("notes_md") or (
        f"# {new_exp}\n\n**Hypothesis:** {plan.get('hypothesis')}\n\n"
        f"**Change:** {plan.get('one_change')}\n"
    )
    (exp_dir / "NOTES.md").write_text(notes if notes.endswith("\n") else notes + "\n", encoding="utf-8")


def clear_qwen_session() -> None:
    """Force a brand-new Qwen Code session (no long chat resume)."""
    chats = QWEN_PROJECT / "chats"
    mem = QWEN_PROJECT / "memory"
    if chats.exists():
        shutil.rmtree(chats, ignore_errors=True)
    if mem.exists():
        shutil.rmtree(mem, ignore_errors=True)
    QWEN_PROJECT.mkdir(parents=True, exist_ok=True)


def ollama_ready(model: str) -> bool:
    try:
        req = urllib.request.Request("http://127.0.0.1:11434/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
        names = [m.get("name") for m in data.get("models") or []]
        return model in names or any(model.split(":")[0] in (n or "") for n in names)
    except Exception:
        return False


def run_executor(
    plan: dict,
    new_exp: str,
    *,
    model: str,
    base_url: str,
    agent_bin: str,
    timeout: int,
) -> None:
    """Fresh Qwen session implements plan (code path)."""
    clear_qwen_session()
    if not ollama_ready(model) and "11434" in base_url:
        raise RuntimeError(f"Ollama model not ready: {model}. Run: ollama pull {model}")

    checklist = "\n".join(f"- {x}" for x in (plan.get("executor_checklist") or []))
    prompt = f"""FRESH SESSION. You are the EXECUTOR only (Qwen). Planner already ran (Cursor Fable).

Read these files first — they are the whole task:
1. outputs/NEXT_STRATEGY.md   ← primary human plan
2. outputs/iteration_plan.json
3. exps/{new_exp}/config.json (seeded from plan — update if needed)
4. exps/{new_exp}/NOTES.md

## Mission
{plan.get('one_change') or plan.get('hypothesis')}

needs_code={plan.get('needs_code')}
code_instructions:
{plan.get('code_instructions') or '(follow NEXT_STRATEGY.md)'}

## Checklist
{checklist}

## Rules
- Implement exactly the one change in NEXT_STRATEGY.md.
- Prefer exps/{new_exp}/ + minimal src/ev_s6e9/** only if the plan requires code.
- Do NOT train full data. Do NOT kaggle submit. Do NOT re-plan STRATEGY.
- When done, stop.
"""
    env = os.environ.copy()
    env.update(
        {
            "OPENAI_API_KEY": "ollama",
            "OLLAMA_API_KEY": "ollama",
            "OPENAI_BASE_URL": base_url.rstrip("/"),
            "OPENAI_MODEL": model,
            "EV_S6E9_ROOT": str(ROOT),
            "QWEN_CODE_SUPPRESS_YOLO_WARNING": "1",
        }
    )
    # New session: no --continue. YOLO for non-interactive writes.
    # Avoid --safe-mode so project QWEN.md can help briefly; plan is short.
    cmd = [agent_bin, "-y", "-m", model, "-o", "text", "-p", prompt]
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"executor_{int(time.time())}.log"
    print(f"executor log → {log_path}", flush=True)
    print(f"executor model={model} base_url={base_url} (fresh session)", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"cmd: {' '.join(cmd)}\nmodel={model}\nbase_url={base_url}\n\n")
        log.flush()
        p = subprocess.Popen(
            cmd, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT, text=True
        )
        try:
            rc = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            # also kill children
            run(["pkill", "-f", "qwen-code"] )
            raise RuntimeError(f"executor timed out after {timeout}s")
    if rc != 0:
        raise RuntimeError(f"executor exited {rc}; see {log_path}")


def run_exp(exp_id: str, timeout: int) -> dict:
    cmd = [py(), str(ROOT / "scripts" / "run_exp.py"), exp_id]
    env = os.environ.copy()
    env["EV_S6E9_ROOT"] = str(ROOT)
    tok = Path.home() / ".kaggle" / "access_token"
    if tok.exists():
        env["KAGGLE_API_TOKEN"] = tok.read_text(encoding="utf-8").strip()
    print(f"$ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=str(ROOT), env=env, text=True, capture_output=True, timeout=timeout)
    (LOG_DIR / "last_train.stdout.txt").write_text(r.stdout or "", encoding="utf-8")
    (LOG_DIR / "last_train.stderr.txt").write_text(r.stderr or "", encoding="utf-8")
    if r.stdout:
        print(r.stdout[-3000:], flush=True)
    if r.returncode != 0:
        print(r.stderr[-2000:] if r.stderr else "", file=sys.stderr)
        raise RuntimeError(f"run_exp failed rc={r.returncode}")
    met_path = EXPS / exp_id / "metrics.json"
    if not met_path.exists():
        raise RuntimeError(f"missing {met_path}")
    return json.loads(met_path.read_text(encoding="utf-8"))


def append_index(row: str) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    if not INDEX.exists():
        INDEX.write_text(
            "# Experiment index\n\n| id | idea | CV | LB | verdict | notes |\n|----|------|----|----|---------|-------|\n",
            encoding="utf-8",
        )
    text = INDEX.read_text(encoding="utf-8")
    if not text.endswith("\n"):
        text += "\n"
    INDEX.write_text(text + row + "\n", encoding="utf-8")


def append_learning(line: str) -> None:
    if not LEARNINGS.exists():
        LEARNINGS.write_text("# LEARNINGS\n\n", encoding="utf-8")
    with LEARNINGS.open("a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")


def git_commit(msg: str) -> str:
    git("add", "-A")
    r = run(["git", "status", "--porcelain"])
    if not (r.stdout or "").strip():
        return head_sha()
    run(["git", "commit", "-m", msg])
    return head_sha()


def git_push(branch: str) -> None:
    r = run(["git", "push", "-u", "origin", branch])
    if r.returncode != 0:
        print(f"push failed:\n{r.stderr or r.stdout}", file=sys.stderr)


def reject_exp(exp_id: str, accept_sha: str, mode: str) -> None:
    exp_dir = EXPS / exp_id
    if mode == "keep":
        cfg_p = exp_dir / "config.json"
        if cfg_p.exists():
            cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
            cfg["status"] = "kill"
            cfg_p.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
        git_commit(f"loop: KILL {exp_id}")
        return
    if accept_sha:
        git("reset", "--hard", accept_sha)
    if exp_dir.exists():
        shutil.rmtree(exp_dir, ignore_errors=True)
    r = run(["git", "status", "--porcelain", "-u"])
    for line in (r.stdout or "").splitlines():
        path = line[3:].strip()
        if not path:
            continue
        top = path.split("/", 1)[0]
        if top in KEEP_ON_CLEAN or path.startswith("exps/exp0000"):
            continue
        p = ROOT / path
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        elif p.exists():
            p.unlink(missing_ok=True)


def run_predict_submit(cfg: dict, do_submit: bool) -> bool:
    args = list(cfg.get("predict_args") or [])
    if not args and cfg.get("strategy"):
        args = ["--strategy", str(cfg["strategy"])]
    env = os.environ.copy()
    env["EV_S6E9_ROOT"] = str(ROOT)
    r = subprocess.run(
        [py(), "-m", "ev_s6e9", "predict", *args],
        cwd=str(ROOT),
        env=env,
        text=True,
        capture_output=True,
    )
    if r.returncode != 0:
        print(f"predict failed: {r.stderr or r.stdout}", file=sys.stderr)
        return False
    if not do_submit:
        return False
    tok = Path.home() / ".kaggle" / "access_token"
    if tok.exists():
        env["KAGGLE_API_TOKEN"] = tok.read_text(encoding="utf-8").strip()
    msg = str(cfg.get("submit_message") or cfg.get("id") or "loop")
    r = subprocess.run(
        [py(), "-m", "ev_s6e9", "submit", "-m", msg],
        cwd=str(ROOT),
        env=env,
        text=True,
        capture_output=True,
    )
    if r.returncode != 0:
        print(f"submit failed: {r.stderr or r.stdout}", file=sys.stderr)
        return False
    print(r.stdout or "", flush=True)
    return True


def main() -> int:
    load_dotenv(HERMES_ENV)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-iters", type=int, default=10)
    ap.add_argument(
        "--plan-backend",
        default=DEFAULT_PLAN_BACKEND,
        choices=["cursor", "api"],
        help="cursor=Cursor Agent Claude Fable; api=OpenRouter/xAI fallback",
    )
    ap.add_argument(
        "--plan-model",
        default=DEFAULT_PLAN_MODEL,
        help="Cursor model id (default claude-fable-5-1-thinking-high) or API model",
    )
    ap.add_argument("--cursor-agent", default=DEFAULT_CURSOR_AGENT)
    ap.add_argument("--planner-timeout", type=int, default=900, help="Cursor/API planner timeout (s)")
    ap.add_argument("--exec-model", default=DEFAULT_EXEC_MODEL)
    ap.add_argument("--exec-base-url", default=DEFAULT_EXEC_BASE)
    ap.add_argument("--agent", default=DEFAULT_AGENT)
    ap.add_argument("--agent-timeout", type=int, default=3600, help="executor timeout (s)")
    ap.add_argument("--train-timeout", type=int, default=10800)
    ap.add_argument("--min-delta", type=float, default=1e-4)
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--branch", default=BRANCH)
    ap.add_argument("--on-reject", choices=["reset", "keep"], default="reset")
    ap.add_argument("--no-executor", action="store_true", help="apply plan config only; skip Qwen")
    ap.add_argument("--force-executor", action="store_true", help="always run Qwen even if needs_code=false")
    ap.add_argument("--dry-run", action="store_true", help="plan only for next exp")
    ap.add_argument("--stop-after-no-improve", type=int, default=8)
    ap.add_argument("--target-cv", type=float, default=0.94672)
    args = ap.parse_args()

    os.chdir(ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    EXPS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if not (EXPS / "exp0000" / "config.json").exists():
        print("missing exps/exp0000", file=sys.stderr)
        return 2

    if args.plan_backend == "cursor":
        cab = args.cursor_agent
        if shutil.which(cab) is None and not Path(cab).exists():
            print(f"Cursor agent not found: {cab}", file=sys.stderr)
            return 2
        print(f"planner ready: Cursor {args.plan_model} via {cab}", flush=True)
    else:
        try:
            base, _, model = resolve_api_planner()
            print(f"planner ready: API {model} @ {base}", flush=True)
        except Exception as e:
            print(f"planner not configured: {e}", file=sys.stderr)
            return 2

    if not args.dry_run and not args.no_executor:
        if shutil.which(args.agent) is None:
            print(f"agent not found: {args.agent}", file=sys.stderr)
            return 2
        if "11434" in args.exec_base_url and not ollama_ready(args.exec_model):
            print(f"pull executor model: ollama pull {args.exec_model}", file=sys.stderr)
            return 2
        print(f"executor ready: {args.exec_model} @ {args.exec_base_url}", flush=True)

    try:
        ensure_branch(args.branch)
    except Exception as e:
        print(f"git: {e}", file=sys.stderr)

    state = State.load(STATE_PATH)
    bcv, bcs, bexp = best_from_exps()
    state.best_cv = max(state.best_cv, bcv, DEFAULT_BEST)
    if not state.best_cv_str or abs(state.best_cv - bcv) < 1e-12:
        state.best_cv_str = bcs or f"{state.best_cv:.5f}"
    state.best_exp = bexp or state.best_exp or "exp0000"
    state.branch = args.branch
    if not state.accept_sha:
        try:
            state.accept_sha = head_sha()
        except Exception:
            pass
    state.save(STATE_PATH)
    print(
        f"branch={args.branch} best_cv={state.best_cv:.5f} best_exp={state.best_exp} "
        f"accept={state.accept_sha[:12]}",
        flush=True,
    )

    for _ in range(args.max_iters):
        state = State.load(STATE_PATH)
        if state.stop == "1":
            print("stop=1")
            break
        state.iteration += 1
        it = state.iteration
        parent = state.best_exp or "exp0000"
        new_id = next_exp_id()
        print(f"\n======== ITERATION {it} {parent} → {new_id} ========", flush=True)

        copy_exp(parent, new_id)
        pre_sha = head_sha()
        try:
            plan = run_planner(
                state,
                new_id,
                parent,
                backend=args.plan_backend,
                model=args.plan_model,
                cursor_bin=args.cursor_agent,
                timeout=args.planner_timeout,
            )
            apply_plan_files(plan, new_id)
            print(
                f"plan: {plan.get('title')} | needs_code={plan.get('needs_code')} | "
                f"backend={plan.get('planner_backend')}",
                flush=True,
            )
        except Exception as e:
            print(f"planner failed: {e}", file=sys.stderr)
            day = datetime.now(timezone.utc).date().isoformat()
            append_learning(f"- {day} {new_id}: PLANNER FAILED — {e}")
            append_index(f"| {new_id} | planner-fail | — | — | kill | {str(e)[:60]} |")
            state.rejects += 1
            state.no_improve_streak += 1
            reject_exp(new_id, state.accept_sha or pre_sha, args.on_reject)
            state.save(STATE_PATH)
            if state.no_improve_streak >= args.stop_after_no_improve:
                break
            continue

        if args.dry_run:
            print(f"dry-run: wrote {NEXT_STRATEGY} / {PLAN_JSON}; stop")
            state.iteration -= 1  # don't count dry-run
            state.save(STATE_PATH)
            return 0

        need_exec = bool(plan.get("needs_code")) or args.force_executor
        if need_exec and not args.no_executor:
            try:
                run_executor(
                    plan,
                    new_id,
                    model=args.exec_model,
                    base_url=args.exec_base_url,
                    agent_bin=args.agent,
                    timeout=args.agent_timeout,
                )
            except Exception as e:
                print(f"executor failed: {e}", file=sys.stderr)
                day = datetime.now(timezone.utc).date().isoformat()
                append_learning(f"- {day} {new_id}: EXECUTOR FAILED — {e}")
                append_index(
                    f"| {new_id} | {str(plan.get('title') or 'exec-fail').replace('|','/')} | — | — | kill | exec: {str(e)[:40]} |"
                )
                state.rejects += 1
                state.no_improve_streak += 1
                learn_txt = LEARNINGS.read_text(encoding="utf-8") if LEARNINGS.exists() else None
                index_txt = INDEX.read_text(encoding="utf-8") if INDEX.exists() else None
                plan_j = PLAN_JSON.read_text(encoding="utf-8") if PLAN_JSON.exists() else None
                plan_m = PLAN_MD.read_text(encoding="utf-8") if PLAN_MD.exists() else None
                reject_exp(new_id, state.accept_sha or pre_sha, args.on_reject)
                if args.on_reject == "reset":
                    if learn_txt:
                        LEARNINGS.write_text(learn_txt, encoding="utf-8")
                    if index_txt:
                        INDEX.write_text(index_txt, encoding="utf-8")
                    if plan_j:
                        PLAN_JSON.write_text(plan_j, encoding="utf-8")
                    if plan_m:
                        PLAN_MD.write_text(plan_m, encoding="utf-8")
                    try:
                        git_commit(f"loop: KILL {new_id} executor-fail (log only)")
                    except Exception:
                        pass
                state.save(STATE_PATH)
                if state.no_improve_streak >= args.stop_after_no_improve:
                    break
                continue
        else:
            print("skip executor (config-only plan)", flush=True)

        try:
            met = run_exp(new_id, args.train_timeout)
        except Exception as e:
            print(f"train failed: {e}", file=sys.stderr)
            day = datetime.now(timezone.utc).date().isoformat()
            append_learning(f"- {day} {new_id}: TRAIN FAILED — {e}")
            append_index(
                f"| {new_id} | {str(plan.get('title') or 'train-fail').replace('|','/')} | — | — | kill | train: {str(e)[:40]} |"
            )
            state.rejects += 1
            state.no_improve_streak += 1
            learn_txt = LEARNINGS.read_text(encoding="utf-8") if LEARNINGS.exists() else None
            index_txt = INDEX.read_text(encoding="utf-8") if INDEX.exists() else None
            reject_exp(new_id, state.accept_sha or pre_sha, args.on_reject)
            if args.on_reject == "reset":
                if learn_txt:
                    LEARNINGS.write_text(learn_txt, encoding="utf-8")
                if index_txt:
                    INDEX.write_text(index_txt, encoding="utf-8")
                try:
                    git_commit(f"loop: KILL {new_id} train-fail (log only)")
                except Exception:
                    pass
            state.save(STATE_PATH)
            if state.no_improve_streak >= args.stop_after_no_improve:
                break
            continue

        mean = float(met["cv_mean"])
        cv_s = str(met.get("cv") or f"{mean:.5f}")
        improved = mean >= state.best_cv + args.min_delta
        print(
            f"CV={cv_s} mean={mean:.8f} best={state.best_cv:.8f} "
            f"delta={mean - state.best_cv:+.8f} keep={improved}",
            flush=True,
        )

        cfg = json.loads((EXPS / new_id / "config.json").read_text(encoding="utf-8"))
        title = str(cfg.get("title") or plan.get("title") or new_id)[:80]
        day = datetime.now(timezone.utc).date().isoformat()

        if improved:
            cfg["status"] = "keep"
            cfg["cv_mean"] = mean
            cfg["cv"] = cv_s
            (EXPS / new_id / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
            met["status"] = "keep"
            (EXPS / new_id / "metrics.json").write_text(json.dumps(met, indent=2) + "\n")
            submitted = False
            if args.submit:
                submitted = run_predict_submit(cfg, do_submit=True)
                if submitted:
                    state.submits += 1
            append_index(
                f"| {new_id} | {title.replace('|','/')} | {cv_s} | "
                f"{'pending' if submitted else '—'} | **keep** | parent={parent} |"
            )
            state.best_cv = mean
            state.best_cv_str = cv_s
            state.best_exp = new_id
            state.accepts += 1
            state.no_improve_streak = 0
            sha = git_commit(f"loop: KEEP {new_id} CV={mean:.5f}" + (" +submit" if submitted else ""))
            state.accept_sha = sha
            if args.push:
                git_push(args.branch)
            print(f"KEEP {new_id} sha={sha[:12]}", flush=True)
            if mean >= args.target_cv:
                print(f"stop: target_cv={args.target_cv}")
                state.save(STATE_PATH)
                break
        else:
            append_index(
                f"| {new_id} | {title.replace('|','/')} | {cv_s} | — | kill | "
                f"best={state.best_cv:.5f} parent={parent} |"
            )
            append_learning(
                f"- {day} {new_id}: {title} → CV {cv_s} vs best {state.best_cv_str} — kill"
            )
            state.rejects += 1
            state.no_improve_streak += 1
            learn_txt = LEARNINGS.read_text(encoding="utf-8") if LEARNINGS.exists() else None
            index_txt = INDEX.read_text(encoding="utf-8") if INDEX.exists() else None
            reject_exp(new_id, state.accept_sha or pre_sha, args.on_reject)
            if args.on_reject == "reset":
                if learn_txt:
                    LEARNINGS.write_text(learn_txt, encoding="utf-8")
                if index_txt:
                    INDEX.write_text(index_txt, encoding="utf-8")
                try:
                    git_commit(f"loop: KILL {new_id} CV={mean:.5f} (log only)")
                except Exception:
                    pass
            if args.push:
                git_push(args.branch)
            print(f"KILL {new_id}", flush=True)
            if state.no_improve_streak >= args.stop_after_no_improve:
                print("stop: no-improve streak")
                state.save(STATE_PATH)
                break

        state.save(STATE_PATH)

    state.save(STATE_PATH)
    print(
        f"done. iters={state.iteration} keep={state.accepts} kill={state.rejects} "
        f"submits={state.submits} best={state.best_cv:.5f} exp={state.best_exp}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
