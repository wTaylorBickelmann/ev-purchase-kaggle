#!/usr/bin/env bash
# Point Qwen Code at llama-server (V4-Flash Q3) while keeping Ollama Qwen as fallback.
set -euo pipefail

MODEL_ID="${1:-deepseek-v4-flash-q3}"
BASE_URL="${2:-http://127.0.0.1:8080/v1}"
SETTINGS="${HOME}/.qwen/settings.json"
BACKUP="${SETTINGS}.bak.$(date +%Y%m%d%H%M%S)"

mkdir -p "${HOME}/.qwen"
if [[ -f "${SETTINGS}" ]]; then
  cp "${SETTINGS}" "${BACKUP}"
  echo "backed up → ${BACKUP}"
fi

python3 - <<'PY' "${SETTINGS}" "${MODEL_ID}" "${BASE_URL}"
import json, sys
from pathlib import Path

path = Path(sys.argv[1])
model = sys.argv[2]
base = sys.argv[3].rstrip("/")
data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

data.setdefault("env", {})
data["env"].update({
    "OLLAMA_API_KEY": "ollama",
    "OPENAI_API_KEY": "ollama",
    # Default interactive stays usable; loop overrides via CLI env.
    "OPENAI_BASE_URL": data.get("env", {}).get("OPENAI_BASE_URL", "http://127.0.0.1:11434/v1"),
})

providers = data.setdefault("modelProviders", {}).setdefault("openai", [])
by_id = {p.get("id"): p for p in providers if isinstance(p, dict)}

def entry(mid: str, name: str, base_url: str, ctx: int = 32768) -> dict:
    return {
        "id": mid,
        "name": name,
        "envKey": "OPENAI_API_KEY",
        "baseUrl": base_url,
        "description": name,
        "generationConfig": {
            "timeout": 900000,
            "maxRetries": 5,
            "contextWindowSize": ctx,
            "samplingParams": {
                "temperature": 0.3,
                "top_p": 0.95,
                "max_tokens": 8192,
            },
        },
    }

by_id.setdefault(
    "qwen3.8:27b-q4_K_M",
    entry("qwen3.8:27b-q4_K_M", "Qwen3.8 27B Q4 (Ollama)", "http://127.0.0.1:11434/v1", 65536),
)
# drop obsolete r1:70b entry if present
by_id.pop("deepseek-r1:70b", None)
by_id[model] = entry(model, f"DeepSeek-V4-Flash Q3 (llama-server)", base, 32768)
data["modelProviders"]["openai"] = list(by_id.values())
data.setdefault("security", {}).setdefault("auth", {})["selectedType"] = "openai"
data.setdefault("tools", {})["approvalMode"] = "yolo"
data.setdefault("model", {})
data["model"]["maxAttempts"] = 8
data["model"]["skipLoopDetection"] = True
# keep default interactive model as qwen unless already set to v4
if data["model"].get("name") in {None, "", "deepseek-r1:70b"}:
    data["model"]["name"] = "qwen3.8:27b-q4_K_M"
data["$version"] = data.get("$version", 4)
path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
print(f"updated {path}")
print(f"  registered model id={model} base={base}")
print(f"  interactive default remains: {data['model'].get('name')}")
PY

echo
echo "Loop usage once GGUF is downloaded:"
echo "  bash scripts/serve_v4_flash_q3.sh"
echo "  python scripts/autoloop.py --model ${MODEL_ID} --base-url ${BASE_URL} --submit --push"
