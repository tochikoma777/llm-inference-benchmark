#!/usr/bin/env bash
# End-to-end demo without a GPU.
#
# Runs the analytic simulator in three serving modes — continuous batching,
# static batching and per-request — then builds one combined report so the
# vLLM-vs-HuggingFace comparison is visible out of the box.
#
#   bash scripts/demo.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

PY="${PYTHON:-python3}"
if [[ -x ".venv/bin/python" ]]; then
  PY=".venv/bin/python"
fi

export LLMBENCH_RESULTS_DIR="${LLMBENCH_RESULTS_DIR:-$PROJECT_ROOT/data/results/demo}"

echo "──────────────────────────────────────────────"
echo " llmbench demo · simulated backends"
echo " results: $LLMBENCH_RESULTS_DIR"
echo "──────────────────────────────────────────────"

"$PY" -m llmbench run -c config/default.yaml \
  --name sim-continuous \
  --tag demo \
  --set backend.serving_mode=continuous_batch

"$PY" -m llmbench run -c config/default.yaml \
  --name sim-static \
  --tag demo \
  --set backend.serving_mode=static_batch \
  --set scenarios.prefix_cache=false

"$PY" -m llmbench run -c config/default.yaml \
  --name sim-per-request \
  --tag demo \
  --set backend.serving_mode=per_request \
  --set scenarios.prefix_cache=false

echo
echo "Building combined report…"
"$PY" -m llmbench report --all --out-dir reports/demo --lang auto

echo
echo "Done."
echo "  HTML:     reports/demo/benchmark_report_*.html"
echo "  Markdown: reports/demo/benchmark_report.generated.md"
echo "  Figures:  reports/demo/figures/"
echo "  Dashboard: llmbench dashboard   (results dir: $LLMBENCH_RESULTS_DIR)"
