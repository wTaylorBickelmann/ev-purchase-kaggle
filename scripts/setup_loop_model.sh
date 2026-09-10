#!/usr/bin/env bash
# Register DeepSeek (and keep Qwen) in ~/.qwen/settings.json for Ollama.
set -euo pipefail

MODEL="${1:-deepseek-r1:70b}"
SETTINGS="${HOME}/.qwen/settings.json"
BACKUP="${SETTINGS}.bak.$(date +%Y%m%d%H%M%S)"

mkdir -p "${HOME}/.qwen"
if [[ -f "${SETTINGS}" ]]; then
  cp "${SETTINGS}" "${BACKUP}"
  echo "backed up → ${BACKUP}"
fi

python3 - <<'PY' "${SETTINGS}" "${MODEL}"
import json, sys
from pathlib import Path

path = Path(sys.argv[1])
model = sys.argv[2]
if path.exists():
    data = json.loads(path.read_text(encoding="utf-8"))
else:
    data = {}

data.setdefault("env", {})
data["env"].update({
    "OLLAMA_API_KEY": "ollama",
    "OPENAI_API_KEY": "ollama",
    "OPENAI_BASE_URL": "http://127.0.0.1:11434/v1",
    # do not force OPENAI_MODEL to DS if user still wants qwen default interactively;
    # autoloop passes -m explicitly.
})

providers = data.setdefault("modelProviders", {}).setdefault("openai", [])
# de-dupe by id
by_id = {p.get("id"): p for p in providers if isinstance(p, dict)}

def entry(mid: str, name: str) -> dict:
    return {
        "id": mid,
        "name": name,
        "envKey": "OLLAMA_API_KEY",
        "baseUrl": "http://127.0.0.1:11434/v1",
        "description": f"Local Ollama {mid}",
        "generationConfig": {
            "timeout": 900000,
            "maxRetries": 5,
            "contextWindowSize": 65536,
            "extra_body": {
                "think": False,
                "options": {
                    "temperature": 0.3,
                    "num_ctx": 32768,
                    "num_predict": 8192,
                },
            },
            "samplingParams": {
                "temperature": 0.3,
                "top_p": 0.9,
                "max_tokens": 8192,
            },
        },
    }

by_id.setdefault("qwen3.8:27b-q4_K_M", entry("qwen3.8:27b-q4_K_M", "Qwen3.8 27B Q4 (Ollama)"))
by_id[model] = entry(model, f"DeepSeek/Ollama ({model})")
data["modelProviders"]["openai"] = list(by_id.values())

data.setdefault("security", {}).setdefault("auth", {})["selectedType"] = "openai"
data.setdefault("tools", {})["approvalMode"] = "yolo"
# keep interactive default as whatever was set; only ensure structure
data.setdefault("model", {}).setdefault("name", model)
data["model"]["maxAttempts"] = 8
data["model"]["skipLoopDetection"] = True
data["$version"] = data.get("$version", 4)

path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
print(f"updated {path} with model id={model}")
PY

echo "Checking ollama…"
if ! ollama list 2>/dev/null | grep -q "${MODEL}"; then
  echo "Model ${MODEL} not pulled yet. Run: ollama pull ${MODEL}"
else
  echo "Model ${MODEL} present."
fi

echo "Done. Test: qwen -m ${MODEL} -p 'reply with ok only'"
