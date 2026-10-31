#!/usr/bin/env bash
# Run the real vLLM + HuggingFace comparison on a GPU machine.
#
#   bash scripts/run_gpu.sh /path/to/Qwen2.5-7B-Instruct
#
# Environment:
#   CONCURRENCY   space-separated levels (default "1 2 4 8")
#   REPEATS       repeats per grid point (default 2)
set -euo pipefail

MODEL_PATH="${1:-${LLMBENCH_MODEL_PATH:-}}"
if [[ -z "$MODEL_PATH" ]]; then
  echo "usage: $0 /path/to/model   (or export LLMBENCH_MODEL_PATH)" >&2
  exit 1
fi
if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Model directory not found: $MODEL_PATH" >&2
  exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

PY="${PYTHON:-python3}"
if [[ -x ".venv/bin/python" ]]; then
  PY=".venv/bin/python"
fi

CONCURRENCY="${CONCURRENCY:-1 2 4 8}"
REPEATS="${REPEATS:-2}"

echo "Model:  $MODEL_PATH"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Levels: $CONCURRENCY (repeats=$REPEATS)"

run_one() {
  local config="$1"
  local name="$2"
  echo
  echo "▶ $name"
  "$PY" -m llmbench run -c "$config" \
    --model-path "$MODEL_PATH" \
    --name "$name" \
    --concurrency $CONCURRENCY \
    --set "scenarios.repeats=$REPEATS"
  sleep 8   # let the driver reclaim GPU memory between engines
}

run_one config/hf.yaml   "hf-static-batching"
run_one config/vllm.yaml "vllm-continuous-batching"

echo
"$PY" -m llmbench report --latest --out-dir reports/gpu
echo
echo "Note: use \`llmbench report --all\` to merge both runs into one comparison."
