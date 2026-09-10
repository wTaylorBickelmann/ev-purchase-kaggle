#!/usr/bin/env bash
# Download Unsloth UD-Q3_K_M GGUF for DeepSeek-V4-Flash-0731 (~128 GB).
set -euo pipefail
export PATH="${HOME}/.local/bin:${PATH}"
export HF_HUB_ENABLE_HF_TRANSFER=1

OUT="${1:-$HOME/Models/deepseek-v4-flash-q3}"
mkdir -p "${OUT}"
echo "→ ${OUT}/UD-Q3_K_M (Unsloth Q3_K_M, ~128GB)"
hf download unsloth/DeepSeek-V4-Flash-0731-GGUF \
  --include 'UD-Q3_K_M/*' \
  --local-dir "${OUT}"

echo
ls -lh "${OUT}/UD-Q3_K_M" || true
echo "Done. Start server: bash scripts/serve_v4_flash_q3.sh"
