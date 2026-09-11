#!/usr/bin/env python3
"""Deotte / BirdCLEF-style score-gated experiment factory.

Fresh agent context every iteration. **Files are memory.**

Cycle:
  1. Read STRATEGY.md, LEARNINGS.md, reports/index.md, last keep exp
  2. Copy exps/expNNNN → expNNNN+1
  3. Agent makes **one** change in the new folder (config / NOTES; shared lib only if required)
  4. python scripts/run_exp.py expNNNN+1  → metrics.json with cv_mean
  5. Keep iff cv_mean >= best + min_delta; else kill (reset tree) + LEARNINGS
  6. Append reports/index.md; optional Kaggle submit on keep

Examples:
  python scripts/autoloop.py --dry-run
  # terminal A: bash scripts/serve_v4_flash_q3.sh
  # terminal B:
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
BRIEF_PATH = OUT / "RUN_BRIEF.md"
LOG_DIR = ROOT / "logs" / "loop"
VENV_PY = ROOT / ".venv" / "bin" / "python"
DEFAULT_MODEL = os.environ.get("EV_LOOP_MODEL", "deepseek-v4-flash-q3")
DEFAULT_BASE_URL = os.environ.get("EV_LOOP_BASE_URL", "http://127.0.0.1:8080/v1")
DEFAULT_AGENT = os.environ.get("EV_LOOP_AGENT", "qwen")
DEFAULT_BEST = 0.94210
BRANCH = "loop/auto"
V4_Q3_DIR = Path.home() / "Models" / "deepseek-v4-flash-q3" / "UD-Q3_K_M"

KEEP_ON_CLEAN = {
    "data",
    ".venv",
    "outputs",
    "logs",
    ".git",
    ".pytest_cache",
    "__pycache__",
    "exps",  # handled explicitly
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
    listed = git("branch", "--list", branch)
    if listed:
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


def best_from_index_and_exps() -> tuple[float, str, str]:
    """Return best_cv, best_cv_str, best_exp_id from non-kill metrics."""
    best = 0.0
    best_s = ""
    best_exp = "exp0000"
    for p in list_exps():
        met = load_metrics(p) or {}
        cfg: dict = {}
        if (p / "config.json").exists():
            cfg = json.loads((p / "config.json").read_text(encoding="utf-8"))
        status = str(met.get("status") or cfg.get("status") or "").lower()
        if status == "kill":
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
    src = EXPS / src_id
    dst = EXPS / dst_id
    if not src.is_dir():
        raise FileNotFoundError(src)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("models", "__pycache__", "*.joblib"))
    # reset runtime artifacts in copy
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
            "hypothesis": cfg.get("hypothesis") or "TBD — agent must set one change",
        }
    )
    for k in ("cv_mean", "cv_std", "cv", "lb"):
        cfg.pop(k, None)
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    notes = dst / "NOTES.md"
    notes.write_text(
        f"# {dst_id}\n\n**Parent:** {src_id}\n\n**Hypothesis:** (agent fills — one change only)\n\n"
        f"**Change log:**\n- copied from {src_id}\n",
        encoding="utf-8",
    )
    return dst


def _tail_lines(path: Path, n: int) -> str:
    if not path.exists():
        return "(missing)"
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) <= n:
        return "\n".join(lines)
    return "\n".join(lines[-n:])


def write_brief(state: State, new_exp: str, parent: str) -> Path:
    """Compact task card — agent must open STRATEGY/LEARNINGS/index on disk."""
    OUT.mkdir(parents=True, exist_ok=True)
    index_tail = _tail_lines(INDEX, 12)
    learnings_tail = _tail_lines(LEARNINGS, 15)
    # queue excerpt only (not full STRATEGY)
    strategy_q = ""
    if STRATEGY.exists():
        body = STRATEGY.read_text(encoding="utf-8")
        if "## Queue" in body:
            strategy_q = body.split("## Queue", 1)[1]
            strategy_q = "## Queue" + strategy_q.split("## Stop", 1)[0]
        strategy_q = strategy_q[:1800]
    parent_cfg = ""
    pcfg = EXPS / parent / "config.json"
    if pcfg.exists():
        parent_cfg = pcfg.read_text(encoding="utf-8")[:1200]
    brief = f"""# RUN BRIEF — {new_exp} ← parent {parent}

Brain: **DeepSeek-V4-Flash-Q3** via llama-server. Files are memory.

## Mission
One change in `exps/{new_exp}/` to beat CV **{state.best_cv_str}** (gate {state.best_cv:.8f}).

## Edit only
- `exps/{new_exp}/config.json`
- `exps/{new_exp}/NOTES.md`
- `src/ev_s6e9/**` only if a new flag/capability is required (minimal)

## Do not
- Train full data / Kaggle submit / edit other keep exps / change folds/metric
- Orchestrator runs: `python scripts/run_exp.py {new_exp}`

## config.json must set
id, title, hypothesis (ONE change), parent, strategy or train_args, folds=5, seed, status=wip

## STRATEGY queue (read full STRATEGY.md on disk)
{strategy_q or "(see STRATEGY.md)"}

## LEARNINGS (tail)
{learnings_tail}

## reports/index.md (tail)
{index_tail}

## Parent config
```json
{parent_cfg}
```

Also read `exps/{parent}/NOTES.md`. Then implement the single change and **stop**.
"""
    BRIEF_PATH.write_text(brief, encoding="utf-8")
    return BRIEF_PATH


def openai_server_ready(base_url: str) -> bool:
    try:
        req = urllib.request.Request(
            base_url.rstrip("/") + "/models",
            headers={"Authorization": "Bearer ollama"},
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return resp.status == 200 and bool(body)
    except Exception:
        return False


def model_ready(model: str, base_url: str) -> tuple[bool, str]:
    if openai_server_ready(base_url):
        return True, f"server ok at {base_url}"
    if model.startswith("deepseek-v4") and V4_Q3_DIR.is_dir():
        shards = list(V4_Q3_DIR.glob("*.gguf"))
        if len(shards) >= 4:
            return False, f"GGUF present; start: bash scripts/serve_v4_flash_q3.sh"
    if shutil.which("ollama"):
        r = run(["ollama", "list"])
        if model in (r.stdout or "") and "11434" in base_url:
            return True, "ollama model present"
    return False, f"Model/server not ready for {model} at {base_url}"


def run_agent(model: str, agent_bin: str, timeout: int, base_url: str, new_exp: str) -> None:
    """Drive coding agent against local DeepSeek (OpenAI-compatible llama-server).

    Uses Qwen Code CLI as the *harness* only; model id + base_url must be DeepSeek.
    --safe-mode skips auto-injected project context that blew past ctx limits.
    """
    prompt = (
        f"You are coding against DeepSeek-V4-Flash via local llama-server.\n"
        f"Read outputs/RUN_BRIEF.md then STRATEGY.md LEARNINGS.md reports/index.md "
        f"and exps parent notes. Implement ONE change for {new_exp} in exps/{new_exp}/ only. "
        f"Do not train full data. Do not submit. Stop when config.json + NOTES.md are done."
    )
    env = os.environ.copy()
    env.update(
        {
            "OPENAI_API_KEY": "ollama",
            "OLLAMA_API_KEY": "ollama",
            "OPENAI_BASE_URL": base_url.rstrip("/"),
            "OPENAI_MODEL": model,
            "EV_S6E9_ROOT": str(ROOT),
            "QWEN_CODE_SUPPRESS_YOLO_WARNING": "1",
            # pin model for any nested tool that reads env
            "QWEN_MODEL": model,
        }
    )
    # -p non-interactive; --safe-mode = no huge repo context dumps;
    # -y (yolo) required: safe-mode disables settings approvalMode=yolo
    cmd = [
        agent_bin,
        "--safe-mode",
        "-y",
        "-m",
        model,
        "-o",
        "text",
        "-p",
        prompt,
    ]
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"agent_{int(time.time())}.log"
    print(f"agent log → {log_path}", flush=True)
    print(f"agent model={model} base_url={base_url} (DeepSeek path)", flush=True)
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
            raise RuntimeError(f"agent timed out after {timeout}s")
    if rc != 0:
        raise RuntimeError(f"agent exited {rc}; see {log_path}")


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
        # mark kill but keep folder for forensics
        cfg_p = exp_dir / "config.json"
        if cfg_p.exists():
            cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
            cfg["status"] = "kill"
            cfg_p.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
        git_commit(f"loop: KILL {exp_id}")
        return
    # reset code to last accept, drop failed exp dir
    if accept_sha:
        git("reset", "--hard", accept_sha)
    if exp_dir.exists():
        # if reset didn't remove untracked exp
        shutil.rmtree(exp_dir, ignore_errors=True)
    # clean other untracked junk except protected
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
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-iters", type=int, default=10)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--agent", default=DEFAULT_AGENT)
    ap.add_argument("--agent-timeout", type=int, default=7200)
    ap.add_argument("--train-timeout", type=int, default=10800)
    ap.add_argument("--min-delta", type=float, default=1e-4)
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--branch", default=BRANCH)
    ap.add_argument("--on-reject", choices=["reset", "keep"], default="reset")
    ap.add_argument("--no-agent", action="store_true", help="use existing WIP exp (latest) without agent")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stop-after-no-improve", type=int, default=8)
    ap.add_argument("--target-cv", type=float, default=0.94672)
    args = ap.parse_args()

    os.chdir(ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    EXPS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if not (EXPS / "exp0000" / "config.json").exists():
        print("missing exps/exp0000 — seed baseline first", file=sys.stderr)
        return 2

    if not args.dry_run and not args.no_agent:
        if shutil.which(args.agent) is None:
            print(f"agent not found: {args.agent}", file=sys.stderr)
            return 2
        ok, hint = model_ready(args.model, args.base_url)
        print(hint, flush=True)
        if not ok:
            return 2

    try:
        ensure_branch(args.branch)
    except Exception as e:
        print(f"git: {e}", file=sys.stderr)

    state = State.load(STATE_PATH)
    bcv, bcs, bexp = best_from_index_and_exps()
    state.best_cv = max(state.best_cv, bcv, DEFAULT_BEST)
    if not state.best_cv_str or state.best_cv == bcv:
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
            print("stop=1 in loop_state.json")
            break
        state.iteration += 1
        it = state.iteration
        parent = state.best_exp or "exp0000"
        new_id = next_exp_id()
        print(f"\n======== ITERATION {it} {parent} → {new_id} ========", flush=True)

        if args.no_agent:
            # assume latest wip exists
            exps = list_exps()
            new_id = exps[-1].name if exps else new_id
        else:
            copy_exp(parent, new_id)
            write_brief(state, new_id, parent)
            print(f"wrote {BRIEF_PATH} and {EXPS / new_id}", flush=True)

        if args.dry_run:
            write_brief(state, new_id, parent)
            print("dry-run: stop after copy+brief")
            state.save(STATE_PATH)
            return 0

        pre_sha = head_sha()
        try:
            if not args.no_agent:
                run_agent(args.model, args.agent, args.agent_timeout, args.base_url, new_id)
            # ensure config id
            cfg_p = EXPS / new_id / "config.json"
            cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
            cfg["id"] = new_id
            cfg["parent"] = parent
            cfg_p.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
            met = run_exp(new_id, args.train_timeout)
        except Exception as e:
            print(f"agent/train failed: {e}", file=sys.stderr)
            day = datetime.now(timezone.utc).date().isoformat()
            append_learning(f"- {day} {new_id}: FAILED — {e}")
            append_index(
                f"| {new_id} | failed | — | — | kill | {str(e)[:60]} |"
            )
            state.rejects += 1
            state.no_improve_streak += 1
            reject_exp(new_id, state.accept_sha or pre_sha, args.on_reject)
            state.save(STATE_PATH)
            if args.push and args.on_reject == "keep":
                git_push(args.branch)
            if state.no_improve_streak >= args.stop_after_no_improve:
                print("stop: no-improve streak")
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
        title = str(cfg.get("title") or cfg.get("hypothesis") or new_id)[:80]
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
            (OUT / "best_cv.json").write_text(
                json.dumps({"exp": new_id, "mean": mean, "cv": cv_s}, indent=2) + "\n"
            )
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
            # preserve LEARNINGS + index even on reset: copy out, reset, copy back
            learn_txt = LEARNINGS.read_text(encoding="utf-8") if LEARNINGS.exists() else None
            index_txt = INDEX.read_text(encoding="utf-8") if INDEX.exists() else None
            reject_exp(new_id, state.accept_sha or pre_sha, args.on_reject)
            if args.on_reject == "reset":
                if learn_txt:
                    LEARNINGS.write_text(learn_txt, encoding="utf-8")
                if index_txt:
                    INDEX.write_text(index_txt, encoding="utf-8")
                # commit learnings/index so they persist
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
