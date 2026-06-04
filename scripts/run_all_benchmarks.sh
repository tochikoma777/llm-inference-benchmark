#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "LLM Inference Benchmark - Full Pipeline"
echo "Start: $(date)"
echo "=========================================="

# Phase 1
echo -e "\n[Phase 1/5] Single Request Baseline"
bash scripts/run_baseline.sh

# Phase 2
echo -e "\n[Phase 2/5] Core Mechanism Validation"
bash scripts/run_phase2.sh

# Phase 3
echo -e "\n[Phase 3/5] Systematic Evaluation"
bash scripts/run_phase3.sh

# Phase 4
echo -e "\n[Phase 4/5] Scheduler Simulation"
python src/scheduler_sim/benchmark.py

# Phase 5
echo -e "\n[Phase 5/5] Generate Final Report Figures"
python src/utils/report_generator.py

echo ""
echo "=========================================="
echo "All benchmarks completed!"
echo "Results: data/results/"
echo "Figures: reports/figures/"
echo "Report:  reports/benchmark_report.md"
echo "End: $(date)"
echo "=========================================="