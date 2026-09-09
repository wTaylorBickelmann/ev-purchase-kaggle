"""Append-only EXPERIMENTS.md. Never rewrite old chunks."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from ev_s6e9.paths import EXPERIMENTS_MD

HEADER = """# Experiments

Append-only running log for playground-series-s6e9. **Newest entries at the bottom.**

Append a chunk after every attempt (or let `python -m ev_s6e9 train` do it). Keep it terse. Do not edit old entries.

Template:

```
### YYYY-MM-DD — <model / features / key params>
- CV: <mean AUC ± std>
- LB: <score or —>
- Takeaway: <one line>
```

---

"""


def format_chunk(
    title: str,
    cv: str,
    *,
    lb: str = "—",
    takeaway: str = "",
    day: str | None = None,
) -> str:
    day = day or datetime.now(timezone.utc).date().isoformat()
    takeaway = takeaway or "auto-logged from train"
    return f"### {day} — {title}\n- CV: {cv}\n- LB: {lb}\n- Takeaway: {takeaway}"


def append_chunk(chunk: str, path: Path | None = None) -> Path:
    """Append `chunk` to EXPERIMENTS.md. Creates the file with the header if missing."""
    path = path or EXPERIMENTS_MD
    body = chunk.strip() + "\n\n"
    if not path.exists() or path.read_text(encoding="utf-8").strip() == "":
        path.write_text(HEADER + body, encoding="utf-8")
        return path
    existing = path.read_text(encoding="utf-8")
    prefix = "" if existing.endswith("\n") else "\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(prefix + body)
    return path
