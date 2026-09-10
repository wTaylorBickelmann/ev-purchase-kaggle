#!/usr/bin/env bash
# Fast multi-connection download of V4-Flash Q3 (aria2). Falls back to curl.
set -euo pipefail
ROOT="${HOME}/Models/deepseek-v4-flash-q3"
OUT="${ROOT}/UD-Q3_K_M"
LOG="${ROOT}/download.log"
mkdir -p "${OUT}"
export PATH="${HOME}/.local/bin:${PATH}"
export HF_HUB_DISABLE_XET=1

python3 - <<'PY'
import json, os
from pathlib import Path
os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import hf_hub_url, get_hf_file_metadata
repo = "unsloth/DeepSeek-V4-Flash-0731-GGUF"
files = [
    "UD-Q3_K_M/DeepSeek-V4-Flash-0731-UD-Q3_K_M-00001-of-00004.gguf",
    "UD-Q3_K_M/DeepSeek-V4-Flash-0731-UD-Q3_K_M-00002-of-00004.gguf",
    "UD-Q3_K_M/DeepSeek-V4-Flash-0731-UD-Q3_K_M-00003-of-00004.gguf",
    "UD-Q3_K_M/DeepSeek-V4-Flash-0731-UD-Q3_K_M-00004-of-00004.gguf",
]
out = {}
for f in files:
    m = get_hf_file_metadata(hf_hub_url(repo, f))
    out[f] = {"url": m.location or hf_hub_url(repo, f), "size": int(m.size)}
root = Path.home() / "Models/deepseek-v4-flash-q3"
root.mkdir(parents=True, exist_ok=True)
(root / "shard_urls.json").write_text(json.dumps(out, indent=2) + "\n")
lines = []
for rel, meta in out.items():
    name = rel.split("/")[-1]
    dest = root / "UD-Q3_K_M" / name
    if dest.exists() and dest.stat().st_size >= meta["size"]:
        print(f"skip complete {name}")
        continue
    lines.append(meta["url"])
    lines.append(f"  out={name}")
    lines.append(f"  dir={root / 'UD-Q3_K_M'}")
(root / "aria2_input.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
print(f"aria2 jobs: {len(lines)//3}")
PY

echo "$(date) aria2 download start" | tee -a "${LOG}"
if command -v aria2c >/dev/null 2>&1; then
  # 16 connections per file, 3 files at once; no proxy
  aria2c --input-file="${ROOT}/aria2_input.txt" \
    --continue=true \
    --max-connection-per-server=16 \
    --split=16 \
    --min-split-size=1M \
    --max-concurrent-downloads=3 \
    --file-allocation=none \
    --allow-overwrite=true \
    --auto-file-renaming=false \
    --summary-interval=30 \
    --console-log-level=notice \
    --all-proxy='' \
    --no-proxy='*' \
    2>&1 | tee -a "${LOG}"
else
  echo "aria2c missing; install with: brew install aria2" | tee -a "${LOG}"
  exit 1
fi
echo "$(date) DONE" | tee -a "${LOG}"
du -sh "${OUT}" | tee -a "${LOG}"
ls -lh "${OUT}" | tee -a "${LOG}"
