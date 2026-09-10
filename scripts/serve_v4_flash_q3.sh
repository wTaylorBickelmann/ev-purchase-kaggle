#!/usr/bin/env bash
# Serve DeepSeek-V4-Flash-0731 UD-Q3_K_M via llama-server (OpenAI-compatible).
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-$HOME/Models/deepseek-v4-flash-q3/UD-Q3_K_M}"
# First shard is the entrypoint for split GGUFs
MODEL_PATH="${MODEL_PATH:-}"
if [[ -z "${MODEL_PATH}" ]]; then
  if [[ -d "${MODEL_DIR}" ]]; then
    MODEL_PATH="$(ls "${MODEL_DIR}"/DeepSeek-V4-Flash-0731-UD-Q3_K_M-00001-of-*.gguf 2>/dev/null | head -1 || true)"
  fi
fi
if [[ -z "${MODEL_PATH}" || ! -f "${MODEL_PATH}" ]]; then
  echo "Model not found under ${MODEL_DIR}"
  echo "Download first:"
  echo "  hf download unsloth/DeepSeek-V4-Flash-0731-GGUF --include 'UD-Q3_K_M/*' --local-dir \$HOME/Models/deepseek-v4-flash-q3"
  exit 1
fi

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
CTX="${CTX:-65536}"
THREADS="${THREADS:-0}"   # 0 = llama default
MODEL_NAME="${MODEL_NAME:-deepseek-v4-flash-q3}"

# Metal on by default for Apple Silicon
EXTRA=( -ngl 99 )
if [[ "${CPU_ONLY:-0}" == "1" ]]; then
  EXTRA=( -ngl 0 )
fi

echo "Serving: ${MODEL_PATH}"
echo "OpenAI base: http://${HOST}:${PORT}/v1  model id: ${MODEL_NAME}"
echo "Ctrl-C to stop."

exec llama-server \
  -m "${MODEL_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --alias "${MODEL_NAME}" \
  -c "${CTX}" \
  -fa on \
  "${EXTRA[@]}" \
  --jinja \
  "$@"
