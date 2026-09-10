#!/usr/bin/env python3
"""Poll Kaggle submissions until latest has a public score."""
from __future__ import annotations

import csv
import io
import os
import re
import subprocess
import sys
import time
from pathlib import Path

TOKEN_PATH = Path.home() / ".kaggle" / "access_token"
COMP = "playground-series-s6e9"


def ensure_token() -> None:
    tok = TOKEN_PATH.read_text(encoding="utf-8").strip()
    os.environ["KAGGLE_API_TOKEN"] = tok


def submissions() -> list[dict]:
    r = subprocess.run(
        ["kaggle", "competitions", "submissions", "-c", COMP, "-v"],
        capture_output=True,
        text=True,
        check=False,
    )
    if r.returncode != 0:
        print(r.stderr or r.stdout, file=sys.stderr)
        return []
    if not r.stdout.strip():
        return []
    return list(csv.DictReader(io.StringIO(r.stdout)))


def score_of(row: dict) -> str | None:
    for k in ("publicScore", "PublicScore", "score", "Score"):
        v = (row.get(k) or "").strip()
        if re.match(r"^\d+\.\d+$", v):
            return v
    return None


def main() -> int:
    ensure_token()
    deadline = time.time() + 240
    latest = None
    while time.time() < deadline:
        rows = submissions()
        if not rows:
            print("no rows yet")
            time.sleep(10)
            continue
        latest = rows[0]
        keys = list(latest.keys())
        print("keys", keys)
        print({k: latest.get(k) for k in keys})
        sc = score_of(latest)
        status = (latest.get("status") or latest.get("Status") or "").strip()
        print(f"status={status!r} score={sc!r}")
        if sc:
            Path("/tmp/kaggle_lb_score.txt").write_text(sc, encoding="utf-8")
            print("LB", sc)
            return 0
        time.sleep(12)
    print("timeout; last=", latest)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
