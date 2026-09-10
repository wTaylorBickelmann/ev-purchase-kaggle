#!/usr/bin/env python3
"""Run one experiment folder: exps/expNNNN/config.json → train → metrics.json.

Usage:
  python scripts/run_exp.py exp0001
  python scripts/run_exp.py exps/exp0001 --synth
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPS = ROOT / "exps"
OUT = ROOT / "outputs"
VENV_PY = ROOT / ".venv" / "bin" / "python"


def py() -> str:
    return str(VENV_PY) if VENV_PY.exists() else sys.executable


def resolve_exp(name: str) -> Path:
    p = Path(name)
    if p.is_dir() and (p / "config.json").exists():
        return p.resolve()
    cand = EXPS / name
    if cand.is_dir() and (cand / "config.json").exists():
        return cand.resolve()
    # allow exp0001 without path
    if not name.startswith("exp"):
        cand = EXPS / f"exp{name}"
        if cand.is_dir():
            return cand.resolve()
    raise FileNotFoundError(f"experiment not found: {name} (expected exps/expNNNN/config.json)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("exp", help="exp id or path (e.g. exp0001)")
    ap.add_argument("--synth", action="store_true", help="tiny synthetic train (smoke)")
    ap.add_argument("--no-log-experiments-md", action="store_true")
    args = ap.parse_args()

    exp_dir = resolve_exp(args.exp)
    cfg_path = exp_dir / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    os.environ["EV_S6E9_ROOT"] = str(ROOT)
    train_args = list(cfg.get("train_args") or [])
    if not train_args and cfg.get("strategy"):
        train_args = ["--strategy", str(cfg["strategy"])]
    if cfg.get("note") and "--note" not in train_args:
        train_args += ["--note", str(cfg["note"])]
    elif cfg.get("hypothesis") and "--note" not in train_args:
        train_args += ["--note", f"{cfg.get('id','exp')}: {cfg['hypothesis'][:80]}"]
    if args.synth and "--synth" not in train_args:
        train_args.append("--synth")
    if args.no_log_experiments_md and "--no-log" not in train_args:
        train_args.append("--no-log")
    # folds/seed if present and not already in args
    if cfg.get("folds") is not None and "--folds" not in train_args:
        train_args += ["--folds", str(int(cfg["folds"]))]
    if cfg.get("seed") is not None and "--seed" not in train_args:
        train_args += ["--seed", str(int(cfg["seed"]))]

    cmd = [py(), "-m", "ev_s6e9", "train", *train_args]
    print(" ".join(cmd), flush=True)
    env = os.environ.copy()
    tok = Path.home() / ".kaggle" / "access_token"
    if tok.exists():
        env["KAGGLE_API_TOKEN"] = tok.read_text(encoding="utf-8").strip()

    r = subprocess.run(cmd, cwd=str(ROOT), env=env, text=True)
    if r.returncode != 0:
        return r.returncode

    cv_path = OUT / "cv.json"
    if not cv_path.exists():
        print(f"missing {cv_path}", file=sys.stderr)
        return 2
    cv = json.loads(cv_path.read_text(encoding="utf-8"))
    metrics = {
        "id": cfg.get("id") or exp_dir.name,
        "cv_mean": float(cv["mean"]),
        "cv_std": float(cv.get("std") or 0.0),
        "cv": cv.get("cv") or f"{cv['mean']:.5f}",
        "fold_aucs": cv.get("fold_aucs"),
        "strategy": cv.get("strategy") or cfg.get("strategy"),
        "params": cv.get("params"),
        "status": "scored",
    }
    (exp_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    # snapshot oof next to exp for stacking later
    oof = OUT / "oof.csv"
    if oof.exists():
        shutil.copy2(oof, exp_dir / "oof.csv")
    shutil.copy2(cv_path, exp_dir / "cv.json")

    # mirror key fields onto config
    cfg["cv_mean"] = metrics["cv_mean"]
    cfg["cv_std"] = metrics["cv_std"]
    cfg["cv"] = metrics["cv"]
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")

    print(f"cv_score={metrics['cv_mean']:.8f}")
    print(f"wrote {exp_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
