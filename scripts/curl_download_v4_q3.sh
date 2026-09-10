#!/usr/bin/env bash
# Direct curl download of V4-Flash Q3 shards (bypasses FreedomProxy/Xet stalls).
set -euo pipefail

ROOT="${HOME}/Models/deepseek-v4-flash-q3"
OUT="${ROOT}/UD-Q3_K_M"
LOG="${ROOT}/download.log"
URLS_JSON="${ROOT}/shard_urls.json"
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
    u = hf_hub_url(repo, f)
    m = get_hf_file_metadata(u)
    out[f] = {"url": m.location or u, "size": int(m.size)}
    print(f"{f} -> {m.size/1e9:.2f} GB")
Path(os.path.expanduser("~/Models/deepseek-v4-flash-q3/shard_urls.json")).write_text(
    json.dumps(out, indent=2) + "\n"
)
PY

echo "$(date) curl multi-shard start" | tee -a "${LOG}"

python3 - <<'PY' > /tmp/v4_shards.tsv
import json
from pathlib import Path
d = json.loads(Path.home().joinpath("Models/deepseek-v4-flash-q3/shard_urls.json").read_text())
for rel, meta in d.items():
    name = rel.split("/")[-1]
    # TSV: name \t size \t url
    print(f"{name}\t{meta['size']}\t{meta['url']}")
PY

while IFS=$'\t' read -r name size url; do
  dest="${OUT}/${name}"
  if [[ -f "${dest}" ]]; then
    sz=$(stat -f%z "${dest}" 2>/dev/null || echo 0)
    if [[ "${sz}" -ge "${size}" ]]; then
      echo "$(date) skip ${name} (${sz} bytes)" | tee -a "${LOG}"
      continue
    fi
  fi
  echo "$(date) curl ${name} expect=${size}" | tee -a "${LOG}"
  # --noproxy avoids FreedomProxy localhost intercept; -C - resumes
  if curl -L --noproxy '*' --retry 30 --retry-delay 5 -C - \
      --connect-timeout 30 \
      -o "${dest}" "${url}" >>"${LOG}" 2>&1; then
    echo "$(date) DONE ${name} $(stat -f%z "${dest}")" | tee -a "${LOG}"
  else
    echo "$(date) FAIL ${name} rc=$?" | tee -a "${LOG}"
  fi
done < /tmp/v4_shards.tsv

echo "$(date) ALL SHARDS ATTEMPTED" | tee -a "${LOG}"
du -sh "${OUT}" | tee -a "${LOG}"
ls -lh "${OUT}" | tee -a "${LOG}"
