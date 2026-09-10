
#!/usr/bin/env python3
"""Download Unsloth V4-Flash UD-Q3_K_M shards with resume + progress log."""
from __future__ import annotations
import os, sys, time
from pathlib import Path

os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

from huggingface_hub import hf_hub_download

REPO = "unsloth/DeepSeek-V4-Flash-0731-GGUF"
OUT = Path(os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/Models/deepseek-v4-flash-q3"))
FILES = [
    "UD-Q3_K_M/DeepSeek-V4-Flash-0731-UD-Q3_K_M-00001-of-00004.gguf",
    "UD-Q3_K_M/DeepSeek-V4-Flash-0731-UD-Q3_K_M-00002-of-00004.gguf",
    "UD-Q3_K_M/DeepSeek-V4-Flash-0731-UD-Q3_K_M-00003-of-00004.gguf",
    "UD-Q3_K_M/DeepSeek-V4-Flash-0731-UD-Q3_K_M-00004-of-00004.gguf",
]
LOG = OUT / "download.log"
OUT.mkdir(parents=True, exist_ok=True)

def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")

log(f"start OUT={OUT}")
for i, rel in enumerate(FILES, 1):
    dest = OUT / rel
    if dest.exists() and dest.stat().st_size > 1_000_000:
        # header shard is ~5MB; large shards should be tens of GB
        sz = dest.stat().st_size
        if i == 1 or sz > 10_000_000_000:
            log(f"[{i}/4] skip existing {rel} ({sz/1e9:.2f} GB)")
            continue
    log(f"[{i}/4] downloading {rel} ...")
    t0 = time.time()
    path = hf_hub_download(
        repo_id=REPO,
        filename=rel,
        local_dir=str(OUT),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    dt = time.time() - t0
    sz = Path(path).stat().st_size
    log(f"[{i}/4] done {path} size={sz/1e9:.2f} GB in {dt/60:.1f} min")
log("ALL DONE")
