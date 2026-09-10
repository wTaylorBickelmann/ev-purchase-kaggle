#!/usr/bin/env python3
"""Autonomous Kaggle experiment loop (local model, CV-gated submit).

Orchestrator only — coding is delegated to Qwen Code CLI → Ollama.

Each iteration:
  1. Write outputs/RUN_BRIEF.md from current best + experiment history
  2. Run coding agent (one experiment)
  3. Agent must write outputs/next_experiment.json
  4. Run train command from that JSON
  5. If CV beats best + min_delta → accept, commit, push, optional submit
  6. Else → reject (reset --hard to last accept by default)

Examples:
  python scripts/autoloop.py --dry-run
  python scripts/autoloop.py --max-iters 1 --no-agent   # train whatever next_experiment.json says
  python scripts/autoloop.py --max-iters 20 --submit --push
  python scripts/autoloop.py --model deepseek-r1:70b --max-iters 50 --submit --push

Env:
  EV_LOOP_MODEL   default Ollama model tag
  EV_LOOP_AGENT   agent binary (default: qwen)
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
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
STATE_PATH = OUT / "loop_state.json"
BRIEF_PATH = OUT / "RUN_BRIEF.md"
NEXT_PATH = OUT / "next_experiment.json"
CV_JSON = OUT / "cv.json"
EXPERIMENTS = ROOT / "EXPERIMENTS.md"
LOG_DIR = ROOT / "logs" / "loop"
VENV_PY = ROOT / ".venv" / "bin" / "python"
DEFAULT_MODEL = os.environ.get("EV_LOOP_MODEL", "deepseek-r1:70b")
DEFAULT_AGENT = os.environ.get("EV_LOOP_AGENT", "qwen")
DEFAULT_BEST = 0.94210  # current deotte CV floor if log empty
BRANCH = "loop/auto"

# paths we never wipe on reject
KEEP_ON_CLEAN = {
    "data",
    ".venv",
    "outputs",
    "logs",
    ".git",
    ".pytest_cache",
    "__pycache__",
}


@dataclass
class State:
    best_cv: float
    best_cv_str: str
    accept_sha: str
    iteration: int
    accepts: int
    rejects: int
    submits: int
    no_improve_streak: int
    branch: str
    updated_at: str

    @classmethod
    def load(cls, path: Path) -> State:
        if path.exists():
            d = json.loads(path.read_text(encoding="utf-8"))
            return cls(**d)
        return cls(
            best_cv=0.0,
            best_cv_str="",
            accept_sha="",
            iteration=0,
            accepts=0,
            rejects=0,
            submits=0,
            no_improve_streak=0,
            branch=BRANCH,
            updated_at="",
        )

    def save(self, path: Path) -> None:
        self.updated_at = datetime.now(timezone.utc).isoformat()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")


def py() -> str:
    return str(VENV_PY) if VENV_PY.exists() else sys.executable


def run(
    cmd: list[str] | str,
    *,
    cwd: Path | None = None,
    env: dict | None = None,
    timeout: int | None = None,
    check: bool = False,
    shell: bool = False,
) -> subprocess.CompletedProcess[str]:
    print(f"$ {cmd if isinstance(cmd, str) else ' '.join(cmd)}", flush=True)
    return subprocess.run(
        cmd,
        cwd=str(cwd or ROOT),
        env=env,
        timeout=timeout,
        check=check,
        shell=shell,
        text=True,
        capture_output=True,
    )


def git(*args: str, check: bool = True) -> str:
    r = run(["git", *args], check=False)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed:\n{r.stderr or r.stdout}")
    return (r.stdout or "").strip()


def ensure_repo_branch(branch: str) -> None:
    cur = git("rev-parse", "--abbrev-ref", "HEAD")
    branches = git("branch", "--list", branch)
    if cur == branch:
        return
    if branches:
        git("checkout", branch)
    else:
        git("checkout", "-b", branch)


def head_sha() -> str:
    return git("rev-parse", "HEAD")


def parse_best_cv_from_experiments(text: str) -> tuple[float, str]:
    """Return (max CV mean, raw cv string) from EXPERIMENTS.md chunks."""
    best = 0.0
    best_s = ""
    for m in re.finditer(
        r"^- CV:\s*([0-9]+\.[0-9]+)(?:\s*±\s*[0-9]+\.[0-9]+)?",
        text,
        flags=re.M,
    ):
        v = float(m.group(1))
        if v > best:
            best = v
            best_s = m.group(0).split(":", 1)[1].strip()
    return best, best_s


def read_cv_json(path: Path = CV_JSON) -> tuple[float, str, dict]:
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; train did not write cv.json")
    d = json.loads(path.read_text(encoding="utf-8"))
    mean = float(d["mean"])
    cv_s = str(d.get("cv") or f"{mean:.5f}")
    return mean, cv_s, d


def init_state(state: State) -> State:
    exp_best, exp_s = (0.0, "")
    if EXPERIMENTS.exists():
        exp_best, exp_s = parse_best_cv_from_experiments(EXPERIMENTS.read_text(encoding="utf-8"))
    file_best = 0.0
    file_s = ""
    if CV_JSON.exists():
        try:
            file_best, file_s, _ = read_cv_json()
        except Exception:
            pass
    best = max(state.best_cv, exp_best, file_best, DEFAULT_BEST if state.best_cv <= 0 else 0.0)
    # prefer labeled string from whichever source matched best
    if abs(best - exp_best) < 1e-12 and exp_s:
        best_s = exp_s
    elif abs(best - file_best) < 1e-12 and file_s:
        best_s = file_s
    elif state.best_cv_str:
        best_s = state.best_cv_str
    else:
        best_s = f"{best:.5f}"
    state.best_cv = best
    state.best_cv_str = best_s
    if not state.accept_sha:
        try:
            state.accept_sha = head_sha()
        except Exception:
            state.accept_sha = ""
    return state


def tail_experiments(n: int = 12) -> str:
    if not EXPERIMENTS.exists():
        return "(no EXPERIMENTS.md yet)"
    text = EXPERIMENTS.read_text(encoding="utf-8")
    chunks = re.split(r"(?=^### \d{4}-\d{2}-\d{2})", text, flags=re.M)
    body = [c.strip() for c in chunks if c.strip().startswith("### ")]
    if not body:
        return text[-3000:]
    return "\n\n".join(body[-n:])


def write_brief(state: State, iteration: int) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    strategies = (ROOT / "STRATEGIES.md").read_text(encoding="utf-8") if (ROOT / "STRATEGIES.md").exists() else ""
    cursor = (ROOT / "CURSOR.md").read_text(encoding="utf-8") if (ROOT / "CURSOR.md").exists() else ""
    top20 = ""
    p20 = ROOT / "TOP20_PUBLIC_NOTEBOOKS.md"
    if p20.exists():
        top20 = p20.read_text(encoding="utf-8")[:6000]
    brief = f"""# RUN BRIEF — iteration {iteration}

You are the coding agent in an autonomous Kaggle loop for **playground-series-s6e9** (ROC-AUC).
Work only in this repo. Local DeepSeek via Ollama — no cloud API tokens.

## Goal this iteration
Implement **exactly one** experiment that can beat the current best local CV.

- **Current best CV:** {state.best_cv_str}  (numeric gate: `{state.best_cv:.8f}`)
- **Public reference:** Deotte notebook ~0.94672; our best LB ~0.94182
- **Do not submit to Kaggle yourself.** Orchestrator submits only if CV improves.

## Hard rules
1. One hypothesis only. No multi-unrelated rewrites.
2. Follow `CURSOR.md` (modular classes, readable, thin CLI).
3. If you add a strategy, wire `--strategy` in `src/ev_s6e9/__main__.py` and document it in `STRATEGIES.md`.
4. Prefer real lifts: original-data/recipe features, multi-seed, stacking/blends, notebook ideas from TOP20 — not random `num_leaves` jitter.
5. Do **not** commit secrets or `data/raw` CSVs.
6. Do **not** rewrite old `EXPERIMENTS.md` chunks (train appends new ones).
7. When code is ready, write `outputs/next_experiment.json` (see schema) and stop.

## outputs/next_experiment.json schema
```json
{{
  "title": "short name for logs/commits",
  "hypothesis": "one sentence",
  "strategy": "deotte",
  "train_args": ["--strategy", "deotte", "--note", "your takeaway"],
  "predict_args": ["--strategy", "deotte"],
  "submit_message": "optional message if orchestrator submits"
}}
```
`train_args` are passed to `python -m ev_s6e9 train …`.
Use the project venv mental model: orchestrator runs train, not you (you may smoke-test with `--synth` if cheap).

## Recent experiments
{tail_experiments(12)}

## STRATEGIES.md
{strategies}

## CURSOR.md
{cursor}

## TOP20 notes (truncated)
{top20}
"""
    BRIEF_PATH.write_text(brief, encoding="utf-8")
    return BRIEF_PATH


def default_next_experiment() -> dict:
    return {
        "title": "deotte-baseline-retrain",
        "hypothesis": "Reconfirm deotte CV floor before exploring",
        "strategy": "deotte",
        "train_args": ["--strategy", "deotte", "--note", "autoloop default retrain"],
        "predict_args": ["--strategy", "deotte"],
        "submit_message": "autoloop deotte",
    }


def load_next_experiment() -> dict:
    if not NEXT_PATH.exists():
        raise FileNotFoundError(
            f"missing {NEXT_PATH}; agent must write it before train"
        )
    return json.loads(NEXT_PATH.read_text(encoding="utf-8"))


def run_agent(model: str, agent_bin: str, timeout: int) -> None:
    prompt = (
        f"Read {BRIEF_PATH} and LOOP.md (if present). "
        "Implement ONE experiment end-to-end in this repo. "
        f"Write {NEXT_PATH} when ready to train. "
        "Do not run full (non-synth) train on all data unless very sure it is quick; "
        "orchestrator will train. You may run pytest and synth smoke tests. "
        "Stay on the current git branch; do not force-push."
    )
    env = os.environ.copy()
    env.update(
        {
            "OLLAMA_API_KEY": "ollama",
            "OPENAI_API_KEY": "ollama",
            "OPENAI_BASE_URL": "http://127.0.0.1:11434/v1",
            "OPENAI_MODEL": model,
            "EV_S6E9_ROOT": str(ROOT),
        }
    )
    cmd = [
        agent_bin,
        "-m",
        model,
        "--output-format",
        "text",
        prompt,
    ]
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"agent_{int(time.time())}.log"
    print(f"agent log → {log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"cmd: {' '.join(cmd)}\n\n")
        log.flush()
        p = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            rc = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            raise RuntimeError(f"agent timed out after {timeout}s")
    if rc != 0:
        raise RuntimeError(f"agent exited {rc}; see {log_path}")


def run_train(next_exp: dict, timeout: int) -> subprocess.CompletedProcess[str]:
    args = list(next_exp.get("train_args") or ["--strategy", next_exp.get("strategy", "deotte")])
    cmd = [py(), "-m", "ev_s6e9", "train", *args]
    env = os.environ.copy()
    env["EV_S6E9_ROOT"] = str(ROOT)
    # Kaggle token for anything incidental
    tok = Path.home() / ".kaggle" / "access_token"
    if tok.exists():
        env["KAGGLE_API_TOKEN"] = tok.read_text(encoding="utf-8").strip()
    r = run(cmd, env=env, timeout=timeout, check=False)
    (LOG_DIR / "last_train.stdout.txt").write_text(r.stdout or "", encoding="utf-8")
    (LOG_DIR / "last_train.stderr.txt").write_text(r.stderr or "", encoding="utf-8")
    if r.stdout:
        print(r.stdout[-4000:], flush=True)
    if r.returncode != 0:
        print(r.stderr[-4000:] if r.stderr else "", file=sys.stderr, flush=True)
        raise RuntimeError(f"train failed rc={r.returncode}")
    return r


def run_predict(next_exp: dict, timeout: int = 600) -> None:
    args = list(next_exp.get("predict_args") or [])
    if not args and next_exp.get("strategy"):
        args = ["--strategy", str(next_exp["strategy"])]
    cmd = [py(), "-m", "ev_s6e9", "predict", *args]
    env = os.environ.copy()
    env["EV_S6E9_ROOT"] = str(ROOT)
    r = run(cmd, env=env, timeout=timeout, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"predict failed: {r.stderr or r.stdout}")


def run_submit(message: str, timeout: int = 300) -> None:
    env = os.environ.copy()
    tok = Path.home() / ".kaggle" / "access_token"
    if tok.exists():
        env["KAGGLE_API_TOKEN"] = tok.read_text(encoding="utf-8").strip()
    cmd = [py(), "-m", "ev_s6e9", "submit", "-m", message]
    r = run(cmd, env=env, timeout=timeout, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"submit failed: {r.stderr or r.stdout}")
    print(r.stdout or "", flush=True)


def git_commit_all(msg: str) -> str:
    git("add", "-A")
    # keep data/raw untracked via gitignore; still safe
    r = run(["git", "status", "--porcelain"], check=False)
    if not (r.stdout or "").strip():
        print("nothing to commit", flush=True)
        return head_sha()
    run(["git", "commit", "-m", msg], check=False)
    return head_sha()


def git_push(branch: str) -> None:
    r = run(["git", "push", "-u", "origin", branch], check=False)
    if r.returncode != 0:
        # non-fatal: still local history
        print(f"push failed (continuing):\n{r.stderr or r.stdout}", file=sys.stderr)


def reject_changes(accept_sha: str, mode: str) -> None:
    if mode == "keep":
        git_commit_all("loop: reject (kept for forensics)")
        return
    if not accept_sha:
        print("no accept_sha; skip reset", flush=True)
        return
    git("reset", "--hard", accept_sha)
    # clean untracked code but keep data/venv/outputs/logs
    r = run(["git", "status", "--porcelain", "-u"], check=False)
    for line in (r.stdout or "").splitlines():
        path = line[3:].strip()
        if not path or path.startswith("outputs/") or path.startswith("logs/"):
            continue
        top = path.split("/", 1)[0]
        if top in KEEP_ON_CLEAN:
            continue
        p = ROOT / path
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        elif p.exists():
            p.unlink(missing_ok=True)


def model_ready(model: str) -> bool:
    r = run(["ollama", "list"], check=False)
    return model in (r.stdout or "")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-iters", type=int, default=10)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--agent", default=DEFAULT_AGENT, help="coding CLI binary")
    ap.add_argument("--agent-timeout", type=int, default=3600, help="seconds per agent call")
    ap.add_argument("--train-timeout", type=int, default=7200, help="seconds per full train")
    ap.add_argument("--min-delta", type=float, default=1e-4, help="CV must improve by this much")
    ap.add_argument("--submit", action="store_true", help="Kaggle submit on CV accept")
    ap.add_argument("--push", action="store_true", help="git push after accept/reject commits")
    ap.add_argument("--branch", default=BRANCH)
    ap.add_argument(
        "--on-reject",
        choices=["reset", "keep"],
        default="reset",
        help="reset=hard back to last accept; keep=commit failed attempt",
    )
    ap.add_argument("--no-agent", action="store_true", help="skip agent; use existing next_experiment.json")
    ap.add_argument("--dry-run", action="store_true", help="write brief + print plan; no agent/train")
    ap.add_argument("--stop-after-no-improve", type=int, default=8)
    ap.add_argument("--target-cv", type=float, default=0.94672, help="stop if best reaches this")
    args = ap.parse_args()

    os.chdir(ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if not args.dry_run and not args.no_agent:
        if shutil.which(args.agent) is None:
            print(f"agent binary not found: {args.agent}", file=sys.stderr)
            return 2
        if not model_ready(args.model):
            print(
                f"Ollama model '{args.model}' not in `ollama list`.\n"
                f"  Pull: ollama pull {args.model}\n"
                f"  Or pass --model qwen3.8:27b-q4_K_M while DeepSeek downloads.",
                file=sys.stderr,
            )
            return 2

    try:
        ensure_repo_branch(args.branch)
    except Exception as e:
        print(f"git branch setup: {e}", file=sys.stderr)

    state = init_state(State.load(STATE_PATH))
    state.branch = args.branch
    state.save(STATE_PATH)
    print(
        f"branch={args.branch} best_cv={state.best_cv:.5f} accept_sha={state.accept_sha[:12]}",
        flush=True,
    )

    for _ in range(args.max_iters):
        state.iteration += 1
        it = state.iteration
        print(f"\n======== ITERATION {it} ========", flush=True)
        write_brief(state, it)
        print(f"wrote {BRIEF_PATH}", flush=True)

        if args.dry_run:
            print("dry-run: stopping after brief")
            # seed a sample next_experiment for inspection
            if not NEXT_PATH.exists():
                NEXT_PATH.write_text(json.dumps(default_next_experiment(), indent=2) + "\n")
                print(f"wrote sample {NEXT_PATH}")
            state.save(STATE_PATH)
            return 0

        pre_sha = head_sha()
        try:
            if not args.no_agent:
                run_agent(args.model, args.agent, args.agent_timeout)
            next_exp = load_next_experiment()
        except Exception as e:
            print(f"agent/next_experiment failed: {e}", file=sys.stderr)
            state.rejects += 1
            state.no_improve_streak += 1
            reject_changes(state.accept_sha or pre_sha, args.on_reject)
            state.save(STATE_PATH)
            if state.no_improve_streak >= args.stop_after_no_improve:
                print("stop: no-improve streak")
                break
            continue

        title = str(next_exp.get("title") or f"iter-{it}")
        try:
            run_train(next_exp, args.train_timeout)
            mean, cv_s, payload = read_cv_json()
        except Exception as e:
            print(f"train/cv failed: {e}", file=sys.stderr)
            state.rejects += 1
            state.no_improve_streak += 1
            reject_changes(state.accept_sha or pre_sha, args.on_reject)
            if args.push:
                git_push(args.branch)
            state.save(STATE_PATH)
            if state.no_improve_streak >= args.stop_after_no_improve:
                break
            continue

        improved = mean >= state.best_cv + args.min_delta
        print(
            f"CV={cv_s} mean={mean:.8f} best={state.best_cv:.8f} "
            f"delta={mean - state.best_cv:+.8f} accept={improved}",
            flush=True,
        )

        if improved:
            try:
                run_predict(next_exp)
            except Exception as e:
                print(f"predict failed (still accepting code/CV): {e}", file=sys.stderr)

            submitted = False
            if args.submit:
                msg = str(next_exp.get("submit_message") or f"loop {title} CV={mean:.5f}")
                try:
                    run_submit(msg)
                    state.submits += 1
                    submitted = True
                except Exception as e:
                    print(f"submit failed: {e}", file=sys.stderr)

            state.best_cv = mean
            state.best_cv_str = cv_s
            state.accepts += 1
            state.no_improve_streak = 0
            sha = git_commit_all(
                f"loop: ACCEPT {title} CV={mean:.5f}"
                + (" +submit" if submitted else "")
            )
            state.accept_sha = sha
            # snapshot accepted cv
            (OUT / "best_cv.json").write_text(
                json.dumps({"mean": mean, "cv": cv_s, "payload": payload, "title": title}, indent=2),
                encoding="utf-8",
            )
            if args.push:
                git_push(args.branch)
            print(f"ACCEPTED sha={sha[:12]}", flush=True)
            if mean >= args.target_cv:
                print(f"stop: reached target_cv={args.target_cv}")
                state.save(STATE_PATH)
                break
        else:
            state.rejects += 1
            state.no_improve_streak += 1
            if args.on_reject == "keep":
                git_commit_all(f"loop: REJECT {title} CV={mean:.5f} (best={state.best_cv:.5f})")
            else:
                # commit nothing; reset code. Keep EXPERIMENTS append? reset removes it if committed only —
                # EXPERIMENTS may be dirty: preserve by copying out then back after reset if new chunk added
                exp_before = EXPERIMENTS.read_text(encoding="utf-8") if EXPERIMENTS.exists() else None
                reject_changes(state.accept_sha or pre_sha, "reset")
                # re-append latest train chunk if reset wiped uncommitted EXPERIMENTS changes
                if exp_before and EXPERIMENTS.exists():
                    now = EXPERIMENTS.read_text(encoding="utf-8")
                    if len(exp_before) > len(now):
                        EXPERIMENTS.write_text(exp_before, encoding="utf-8")
                elif exp_before and not EXPERIMENTS.exists():
                    EXPERIMENTS.write_text(exp_before, encoding="utf-8")
            if args.push and args.on_reject == "keep":
                git_push(args.branch)
            print("REJECTED", flush=True)
            if state.no_improve_streak >= args.stop_after_no_improve:
                print("stop: no-improve streak")
                state.save(STATE_PATH)
                break

        state.save(STATE_PATH)

    state.save(STATE_PATH)
    print(
        f"done. iters={state.iteration} accepts={state.accepts} "
        f"rejects={state.rejects} submits={state.submits} best={state.best_cv:.5f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
