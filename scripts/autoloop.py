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
# Durable archives (under reports/ so git tracks them; outputs/ is gitignored)
PLANS_DIR = REPORTS / "plans"
KILLS_DIR = REPORTS / "kills"
EXPERIMENTS_MD = ROOT / "EXPERIMENTS.md"
NEXT_STRATEGY = OUT / "NEXT_STRATEGY.md"  # live pointer for Qwen (also archived each iter)

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
    "reports",  # includes plans/ + kills/ archives
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


def next_exp_id(state: "State | None" = None) -> str:
    """Monotonic exp id — never reuse killed folder numbers (index stays unique)."""
    nums: list[int] = []
    for p in list_exps():
        try:
            nums.append(int(p.name.replace("exp", "")))
        except ValueError:
            continue
    if INDEX.exists():
        for m in re.finditer(r"\bexp(\d{4})\b", INDEX.read_text(encoding="utf-8")):
            nums.append(int(m.group(1)))
    for d in (PLANS_DIR, KILLS_DIR):
        if not d.exists():
            continue
        for child in d.iterdir():
            m = re.match(r"exp(\d{4})", child.name)
            if m:
                nums.append(int(m.group(1)))
    if state is not None:
        nums.append(int(state.iteration))
        nums.append(int(state.accepts) + int(state.rejects))
    n = (max(nums) if nums else -1) + 1
    # skip collisions with live folders
    while (EXPS / f"exp{n:04d}").exists():
        n += 1
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


def _safe_slug(s: str, n: int = 40) -> str:
    s = re.sub(r"[^\w\-]+", "-", (s or "x").strip().lower())
    return (s.strip("-") or "x")[:n]


def snapshot_durable() -> dict[str, str]:
    """Files that must survive git reset --hard on kill."""
    out: dict[str, str] = {}
    for key, path in (
        ("learnings", LEARNINGS),
        ("index", INDEX),
        ("experiments", EXPERIMENTS_MD),
        ("plans_index", PLANS_DIR / "INDEX.md"),
        ("kills_index", KILLS_DIR / "INDEX.md"),
    ):
        if path.exists():
            out[key] = path.read_text(encoding="utf-8")
    return out


def restore_durable(snap: dict[str, str]) -> None:
    mapping = {
        "learnings": LEARNINGS,
        "index": INDEX,
        "experiments": EXPERIMENTS_MD,
        "plans_index": PLANS_DIR / "INDEX.md",
        "kills_index": KILLS_DIR / "INDEX.md",
    }
    for key, path in mapping.items():
        if key not in snap:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(snap[key], encoding="utf-8")


def _append_archive_index(index_path: Path, row: str, header: str) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    if not index_path.exists():
        index_path.write_text(header, encoding="utf-8")
    text = index_path.read_text(encoding="utf-8")
    if not text.endswith("\n"):
        text += "\n"
    index_path.write_text(text + row + "\n", encoding="utf-8")


def archive_plan(exp_id: str, plan: dict | None = None, *, label: str = "") -> Path:
    """Write a permanent copy of the current plan under reports/plans/ (never overwrite)."""
    PLANS_DIR.mkdir(parents=True, exist_ok=True)
    title = ""
    if plan:
        title = str(plan.get("title") or "")
    if not title and PLAN_JSON.exists():
        try:
            title = str(json.loads(PLAN_JSON.read_text(encoding="utf-8")).get("title") or "")
        except Exception:
            title = ""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = _safe_slug(label or title or "plan")
    dest = PLANS_DIR / f"{exp_id}_{ts}_{slug}"
    # uniqueness if same-second collision
    i = 1
    base = dest
    while dest.exists():
        dest = Path(str(base) + f"_{i}")
        i += 1
    dest.mkdir(parents=True, exist_ok=False)

    for src, name in (
        (NEXT_STRATEGY, "NEXT_STRATEGY.md"),
        (PLAN_MD, "ITERATION_PLAN.md"),
        (PLAN_JSON, "iteration_plan.json"),
        (BRIEF_PATH, "RUN_BRIEF.md"),
    ):
        if src.exists():
            shutil.copy2(src, dest / name)
    if plan is not None:
        (dest / "iteration_plan.json").write_text(
            json.dumps(plan, indent=2) + "\n", encoding="utf-8"
        )
        if plan.get("notes_md"):
            (dest / "notes_md.md").write_text(str(plan["notes_md"]), encoding="utf-8")
    meta = {
        "exp_id": exp_id,
        "title": title,
        "archived_at": datetime.now(timezone.utc).isoformat(),
        "label": label,
        "planner_model": (plan or {}).get("planner_model"),
        "planner_backend": (plan or {}).get("planner_backend"),
        "needs_code": (plan or {}).get("needs_code"),
        "dir": str(dest.relative_to(ROOT)),
    }
    (dest / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (PLANS_DIR / "LATEST").write_text(str(dest.relative_to(ROOT)) + "\n", encoding="utf-8")
    _append_archive_index(
        PLANS_DIR / "INDEX.md",
        f"| {ts} | {exp_id} | {title.replace('|', '/') or '—'} | `{dest.relative_to(ROOT)}` |",
        "# Plan archive\n\n| when (UTC) | exp | title | path |\n|------------|-----|-------|------|\n",
    )
    print(f"archived plan → {dest.relative_to(ROOT)}", flush=True)
    return dest


def archive_kill(
    exp_id: str,
    *,
    plan: dict | None = None,
    metrics: dict | None = None,
    reason: str = "kill",
    plan_dir: Path | None = None,
) -> Path:
    """Snapshot killed exp artifacts under reports/kills/ (never overwrite)."""
    KILLS_DIR.mkdir(parents=True, exist_ok=True)
    title = str((plan or {}).get("title") or (metrics or {}).get("id") or exp_id)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = KILLS_DIR / f"{exp_id}_{ts}_{_safe_slug(title)}"
    i = 1
    base = dest
    while dest.exists():
        dest = Path(str(base) + f"_{i}")
        i += 1
    dest.mkdir(parents=True, exist_ok=False)

    exp_dir = EXPS / exp_id
    if exp_dir.is_dir():
        for name in ("config.json", "NOTES.md", "metrics.json", "cv.json", "oof.csv"):
            p = exp_dir / name
            if p.exists():
                shutil.copy2(p, dest / name)
    for src, name in (
        (NEXT_STRATEGY, "NEXT_STRATEGY.md"),
        (PLAN_JSON, "iteration_plan.json"),
        (PLAN_MD, "ITERATION_PLAN.md"),
        (OUT / "cv.json", "outputs_cv.json"),
    ):
        if src.exists():
            shutil.copy2(src, dest / name)
    if plan is not None:
        (dest / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    if metrics is not None:
        (dest / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    if plan_dir and plan_dir.exists():
        (dest / "plan_archive_path.txt").write_text(
            str(plan_dir.relative_to(ROOT)) + "\n", encoding="utf-8"
        )
    meta = {
        "exp_id": exp_id,
        "title": title,
        "reason": reason,
        "archived_at": datetime.now(timezone.utc).isoformat(),
        "cv": (metrics or {}).get("cv"),
        "cv_mean": (metrics or {}).get("cv_mean"),
        "dir": str(dest.relative_to(ROOT)),
    }
    (dest / "kill_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (KILLS_DIR / "LATEST").write_text(str(dest.relative_to(ROOT)) + "\n", encoding="utf-8")
    cv_s = str((metrics or {}).get("cv") or (metrics or {}).get("cv_mean") or "—")
    _append_archive_index(
        KILLS_DIR / "INDEX.md",
        f"| {ts} | {exp_id} | {title.replace('|', '/')} | {cv_s} | {reason} | `{dest.relative_to(ROOT)}` |",
        "# Kill archive\n\n| when (UTC) | exp | title | CV | reason | path |\n|------------|-----|-------|----|--------|------|\n",
    )
    print(f"archived kill → {dest.relative_to(ROOT)}", flush=True)
    return dest


def finalize_kill(
    exp_id: str,
    accept_sha: str,
    on_reject: str,
    *,
    plan: dict | None = None,
    metrics: dict | None = None,
    reason: str = "kill",
    plan_dir: Path | None = None,
    commit_msg: str = "",
    keep_exps: list[str] | None = None,
) -> None:
    """Archive kill bundle, reset tree if needed, restore durable logs, commit."""
    snap = snapshot_durable()
    try:
        kill_dir = archive_kill(
            exp_id, plan=plan, metrics=metrics, reason=reason, plan_dir=plan_dir
        )
        snap = snapshot_durable()  # include new kills index
    except Exception as e:
        print(f"kill archive failed: {e}", file=sys.stderr)
        kill_dir = None
    reject_exp(exp_id, accept_sha, on_reject, keep_exps=keep_exps)
    if on_reject == "reset":
        restore_durable(snap)
        msg = commit_msg or f"loop: KILL {exp_id} ({reason})"
        try:
            git_commit(msg)
        except Exception:
            pass
    if kill_dir:
        print(f"kill forensics: {kill_dir.relative_to(ROOT)}", flush=True)


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
   - Files to touch (≤2 source files preferred)
   - Step-by-step implementation
   - How train_args / config should look
   - What NOT to do (metric/folds/leak)
   - **HARD LIMIT for executor reliability:** keep NEXT_STRATEGY.md under ~80 lines.
     Prefer 3–8 short bullet steps, not 10-page FIND→REPLACE novels.
     Qwen 27B times out on huge plans. If the change needs many edits, split
     into smaller STRATEGY queue items across future iters.
2. `outputs/iteration_plan.json` — machine plan, this exact schema:
```json
{{
  "title": "short-slug",
  "hypothesis": "one sentence",
  "rationale": "why",
  "one_change": "single concrete change",
  "needs_code": true,
  "code_instructions": "≤500 chars summary; detail lives in NEXT_STRATEGY.md",
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
- Prefer **config-only** (`needs_code=false`) when a flag already exists (e.g. `--freq`).
- Do NOT train, submit, or edit src/ yourself — planner only writes plan files.
- Prefer ≤2 files under `src/ev_s6e9/` when code is required.
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
    # Archive any previous live plan before replacing (never lose history)
    if NEXT_STRATEGY.exists() or PLAN_JSON.exists() or PLAN_MD.exists():
        try:
            archive_plan(f"_prior_{new_exp}", None, label="pre-replace")
        except Exception as e:
            print(f"prior plan archive skipped: {e}", file=sys.stderr)
    # Clear live pointers only (archives already saved)
    for p in (PLAN_JSON, PLAN_MD, NEXT_STRATEGY, BRIEF_PATH):
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
        f"Executor: read `outputs/NEXT_STRATEGY.md` + `iteration_plan.json`.\n"
        f"Full archive: see `reports/plans/` (never overwritten).\n",
        encoding="utf-8",
    )
    arch = archive_plan(new_exp, plan, label="cursor")
    plan["archive_dir"] = str(arch.relative_to(ROOT))
    PLAN_JSON.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    # refresh archive copy with archive_dir field
    (arch / "iteration_plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
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
    arch = archive_plan(new_exp, plan, label="api")
    plan["archive_dir"] = str(arch.relative_to(ROOT))
    PLAN_JSON.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    (arch / "iteration_plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
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


def _git_paths_dirty() -> set[str]:
    r = run(["git", "status", "--porcelain", "-u"])
    out: set[str] = set()
    for line in (r.stdout or "").splitlines():
        path = line[3:].strip()
        if " -> " in path:
            path = path.split(" -> ", 1)[-1].strip()
        if path:
            out.add(path)
    return out


def _src_fingerprint() -> str:
    """Cheap hash of tracked src + exp configs we care about changing."""
    parts: list[str] = []
    for pattern in ("src/ev_s6e9/**/*.py", "scripts/*.py"):
        # use git ls-files for stability
        pass
    r = run(["git", "ls-files", "src/ev_s6e9", "scripts/run_exp.py"])
    for rel in (r.stdout or "").splitlines():
        p = ROOT / rel
        if not p.is_file():
            continue
        st = p.stat()
        parts.append(f"{rel}:{st.st_mtime_ns}:{st.st_size}")
    return "|".join(parts)


def run_executor_qwen(
    plan: dict,
    new_exp: str,
    *,
    model: str,
    base_url: str,
    agent_bin: str,
    timeout: int,
) -> None:
    """Fresh Qwen session implements plan (code path). Slim prompt — plan lives on disk."""
    clear_qwen_session()
    if not ollama_ready(model) and "11434" in base_url:
        raise RuntimeError(f"Ollama model not ready: {model}. Run: ollama pull {model}")

    # Keep prompt small. Pasting full FIND/REPLACE novels blows the 3600s budget.
    title = str(plan.get("title") or new_exp)
    one = str(plan.get("one_change") or plan.get("hypothesis") or title)[:400]
    files = plan.get("files_to_edit") or []
    files_s = ", ".join(str(x) for x in files[:12]) if files else "(see NEXT_STRATEGY.md)"
    prompt = f"""FRESH SESSION. You are EXECUTOR only (Qwen). Planner already wrote the plan on disk.

## Read first (do not skip)
1. outputs/NEXT_STRATEGY.md   ← full plan + exact edits
2. outputs/iteration_plan.json
3. exps/{new_exp}/config.json
4. exps/{new_exp}/NOTES.md

## Mission (one change)
{one}

## Touch only
{files_s}
Plus minimal src/ev_s6e9/** if NEXT_STRATEGY requires code.

## Rules
- Follow NEXT_STRATEGY.md exactly. Prefer small search/replace edits.
- Do NOT re-plan. Do NOT full-data train. Do NOT kaggle submit. Do NOT pytest suite.
- Optional: one tiny smoke from NEXT_STRATEGY if it says so (must finish <60s).
- When edits are done, stop. Do not keep iterating.
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
    cmd = [agent_bin, "-y", "-m", model, "-o", "text", "-p", prompt]
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"executor_qwen_{int(time.time())}.log"
    print(f"executor(qwen) log → {log_path}", flush=True)
    print(f"executor model={model} base_url={base_url} timeout={timeout}s", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"cmd: {agent_bin} -y -m {model} -o text -p <slim>\\nmodel={model}\\n\\n")
        log.write(prompt + "\n\n--- output ---\n")
        log.flush()
        p = subprocess.Popen(
            cmd, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT, text=True
        )
        try:
            rc = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            try:
                p.wait(timeout=10)
            except Exception:
                pass
            run(["pkill", "-f", "qwen-code"])
            run(["pkill", "-f", "cli-entry.js"])
            raise RuntimeError(f"qwen executor timed out after {timeout}s; see {log_path}")
    if rc != 0:
        raise RuntimeError(f"qwen executor exited {rc}; see {log_path}")


def run_executor_cursor(
    plan: dict,
    new_exp: str,
    *,
    model: str,
    agent_bin: str,
    timeout: int,
) -> None:
    """Cursor agent implements the plan (fallback when Qwen stalls)."""
    if shutil.which(agent_bin) is None and not Path(agent_bin).exists():
        raise RuntimeError(f"Cursor agent not found: {agent_bin}")
    title = str(plan.get("title") or new_exp)
    one = str(plan.get("one_change") or plan.get("hypothesis") or title)[:500]
    prompt = f"""You are the EXECUTOR for the EV S6E9 experiment factory. Implement ONE planned change.

Read and follow:
1. outputs/NEXT_STRATEGY.md  (authoritative)
2. outputs/iteration_plan.json
3. exps/{new_exp}/config.json and NOTES.md

Mission: {one}

Rules:
- Apply exactly the one change in NEXT_STRATEGY.md.
- Prefer minimal diffs in src/ev_s6e9/** and exps/{new_exp}/**.
- Do NOT full-data train. Do NOT kaggle submit. Do NOT rewrite STRATEGY/LEARNINGS.
- If NEXT_STRATEGY includes a tiny smoke command (<60s), run it once.
- Stop when the code/config matches the plan.
"""
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
    log_path = LOG_DIR / f"executor_cursor_{int(time.time())}.log"
    print(f"executor(cursor) log → {log_path}", flush=True)
    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"cmd: {' '.join(cmd[:8])} ...\\n\\n")
        log.flush()
        p = subprocess.Popen(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, text=True)
        try:
            rc = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            raise RuntimeError(f"cursor executor timed out after {timeout}s; see {log_path}")
    print(f"executor(cursor) done rc={rc} in {time.time()-t0:.0f}s", flush=True)
    if rc != 0:
        raise RuntimeError(f"cursor executor exited {rc}; see {log_path}")


def run_executor(
    plan: dict,
    new_exp: str,
    *,
    model: str,
    base_url: str,
    agent_bin: str,
    timeout: int,
    exec_backend: str = "auto",
    cursor_bin: str = DEFAULT_CURSOR_AGENT,
    cursor_model: str = DEFAULT_PLAN_MODEL,
    cursor_timeout: int | None = None,
) -> str:
    """Run executor. Returns backend used ('qwen'|'cursor').

    auto: try Qwen first; on timeout/fail, one Cursor Fable fallback.
    """
    needs_code = bool(plan.get("needs_code"))
    before_fp = _src_fingerprint()
    before_dirty = _git_paths_dirty()
    c_timeout = int(cursor_timeout or min(timeout, 900))
    backend = (exec_backend or "auto").lower().strip()
    errors: list[str] = []

    def _verify() -> None:
        if not needs_code:
            return
        after_fp = _src_fingerprint()
        after_dirty = _git_paths_dirty()
        changed = after_fp != before_fp or bool(after_dirty - before_dirty)
        # also allow config-only code plans that only touch exp dir
        exp_cfg = EXPS / new_exp / "config.json"
        # needs_code implies src or scripts should move unless plan is config-flag-only
        files = [str(x) for x in (plan.get("files_to_edit") or [])]
        src_expected = any(
            ("src/" in f) or f.endswith(".py") or f.startswith("src") for f in files
        ) or bool(plan.get("code_instructions"))
        if src_expected and after_fp == before_fp:
            raise RuntimeError(
                "executor made no src/ changes (noop). "
                "Plan needs_code=true but fingerprint unchanged."
            )
        if not changed and src_expected:
            raise RuntimeError("executor produced no file changes")

    if backend in ("qwen", "auto"):
        try:
            run_executor_qwen(
                plan,
                new_exp,
                model=model,
                base_url=base_url,
                agent_bin=agent_bin,
                timeout=timeout,
            )
            _verify()
            return "qwen"
        except Exception as e:
            errors.append(f"qwen: {e}")
            print(f"qwen executor failed: {e}", file=sys.stderr)
            if backend == "qwen":
                raise
            print("falling back to Cursor executor…", flush=True)

    if backend in ("cursor", "auto"):
        try:
            run_executor_cursor(
                plan,
                new_exp,
                model=cursor_model,
                agent_bin=cursor_bin,
                timeout=c_timeout,
            )
            _verify()
            return "cursor"
        except Exception as e:
            errors.append(f"cursor: {e}")
            raise RuntimeError("executor failed: " + " | ".join(errors)) from e

    raise RuntimeError(f"unknown exec backend: {exec_backend}")


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


def reject_exp(exp_id: str, accept_sha: str, mode: str, *, keep_exps: list[str] | None = None) -> None:
    exp_dir = EXPS / exp_id
    keep_exps = list(keep_exps or [])
    if "exp0000" not in keep_exps:
        keep_exps.append("exp0000")
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
    if exp_dir.exists() and exp_id not in keep_exps:
        shutil.rmtree(exp_dir, ignore_errors=True)
    r = run(["git", "status", "--porcelain", "-u"])
    for line in (r.stdout or "").splitlines():
        path = line[3:].strip()
        if not path:
            continue
        top = path.split("/", 1)[0]
        if top in KEEP_ON_CLEAN:
            continue
        # never delete keep exp trees
        if any(path == f"exps/{k}" or path.startswith(f"exps/{k}/") for k in keep_exps):
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
    ap.add_argument(
        "--agent-timeout",
        type=int,
        default=int(os.environ.get("EV_LOOP_AGENT_TIMEOUT", "2400")),
        help="Qwen executor timeout (s); default 2400 then Cursor fallback",
    )
    ap.add_argument(
        "--exec-backend",
        choices=["auto", "qwen", "cursor"],
        default=os.environ.get("EV_LOOP_EXEC_BACKEND", "auto"),
        help="auto=Qwen then Cursor fallback on fail/timeout",
    )
    ap.add_argument(
        "--exec-cursor-timeout",
        type=int,
        default=int(os.environ.get("EV_LOOP_EXEC_CURSOR_TIMEOUT", "900")),
        help="Cursor executor fallback timeout (s)",
    )
    ap.add_argument("--train-timeout", type=int, default=10800)
    ap.add_argument("--min-delta", type=float, default=1e-4)
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--branch", default=BRANCH)
    ap.add_argument("--on-reject", choices=["reset", "keep"], default="reset")
    ap.add_argument("--no-executor", action="store_true", help="apply plan config only; skip executor")
    ap.add_argument("--force-executor", action="store_true", help="always run executor even if needs_code=false")
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
        if args.exec_backend in ("qwen", "auto") and shutil.which(args.agent) is None:
            print(f"qwen agent not found: {args.agent}", file=sys.stderr)
            if args.exec_backend == "qwen":
                return 2
            print("will rely on Cursor executor fallback", flush=True)
        if args.exec_backend in ("qwen", "auto") and "11434" in args.exec_base_url:
            if not ollama_ready(args.exec_model):
                print(f"pull executor model: ollama pull {args.exec_model}", file=sys.stderr)
                if args.exec_backend == "qwen":
                    return 2
        print(
            f"executor ready: backend={args.exec_backend} "
            f"qwen={args.exec_model} @ {args.exec_base_url} "
            f"timeout={args.agent_timeout}s cursor_fb={args.exec_cursor_timeout}s",
            flush=True,
        )

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
        new_id = next_exp_id(state)
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
            finalize_kill(
                new_id,
                state.accept_sha or pre_sha,
                args.on_reject,
                reason="planner-fail",
                commit_msg=f"loop: KILL {new_id} planner-fail (log only)",
                keep_exps=[state.best_exp or "exp0000", "exp0000"],
            )
            state.save(STATE_PATH)
            if state.no_improve_streak >= args.stop_after_no_improve:
                break
            continue

        plan_dir = None
        if plan.get("archive_dir"):
            plan_dir = ROOT / str(plan["archive_dir"])

        if args.dry_run:
            print(f"dry-run: wrote {NEXT_STRATEGY} / {PLAN_JSON}; archived under reports/plans/; stop")
            state.iteration -= 1  # don't count dry-run
            state.save(STATE_PATH)
            return 0

        need_exec = bool(plan.get("needs_code")) or args.force_executor
        if need_exec and not args.no_executor:
            try:
                used = run_executor(
                    plan,
                    new_id,
                    model=args.exec_model,
                    base_url=args.exec_base_url,
                    agent_bin=args.agent,
                    timeout=args.agent_timeout,
                    exec_backend=args.exec_backend,
                    cursor_bin=args.cursor_agent,
                    cursor_model=args.plan_model,
                    cursor_timeout=args.exec_cursor_timeout,
                )
                print(f"executor ok via {used}", flush=True)
            except Exception as e:
                print(f"executor failed: {e}", file=sys.stderr)
                day = datetime.now(timezone.utc).date().isoformat()
                append_learning(f"- {day} {new_id}: EXECUTOR FAILED — {e}")
                append_index(
                    f"| {new_id} | {str(plan.get('title') or 'exec-fail').replace('|','/')} | — | — | kill | exec: {str(e)[:40]} |"
                )
                state.rejects += 1
                state.no_improve_streak += 1
                finalize_kill(
                    new_id,
                    state.accept_sha or pre_sha,
                    args.on_reject,
                    plan=plan,
                    reason="executor-fail",
                    plan_dir=plan_dir,
                    commit_msg=f"loop: KILL {new_id} executor-fail (log only)",
                keep_exps=[state.best_exp or "exp0000", "exp0000"],
            )
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
            finalize_kill(
                new_id,
                state.accept_sha or pre_sha,
                args.on_reject,
                plan=plan,
                reason="train-fail",
                plan_dir=plan_dir,
                commit_msg=f"loop: KILL {new_id} train-fail (log only)",
                keep_exps=[state.best_exp or "exp0000", "exp0000"],
            )
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
            # archive keep plan/metrics too
            try:
                keep_arch = archive_plan(new_id, {**plan, "metrics": met, "verdict": "keep"}, label="keep")
                # also copy metrics into plan archive
                (keep_arch / "metrics.json").write_text(json.dumps(met, indent=2) + "\n", encoding="utf-8")
            except Exception as e:
                print(f"keep archive warn: {e}", file=sys.stderr)
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
            finalize_kill(
                new_id,
                state.accept_sha or pre_sha,
                args.on_reject,
                plan=plan,
                metrics=met,
                reason="cv-gate",
                plan_dir=plan_dir,
                commit_msg=f"loop: KILL {new_id} CV={mean:.5f} (log only)",
                keep_exps=[state.best_exp or "exp0000", "exp0000"],
            )
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
