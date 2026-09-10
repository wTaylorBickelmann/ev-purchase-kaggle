#!/usr/bin/env bash
# Download Unsloth UD-Q3_K_M (~128GB). Uses curl --noproxy to avoid local proxy stalls.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
# Prefer repo script; fall back to Models copy
if [[ -f "${ROOT}/curl_download_v4_q3.sh" ]]; then
  exec bash "${ROOT}/curl_download_v4_q3.sh"
fi
exec bash "${HOME}/Models/deepseek-v4-flash-q3/curl_download_shards.sh"
