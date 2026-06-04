#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# 日志配置
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/phase2_full_${TIMESTAMP}.log"

{
    echo "========================================"
    echo "Phase 2: Core Mechanism Validation"
    echo "Log file: $LOG_FILE"
    echo "========================================"

    # 2.1 并发吞吐扫描
    echo ""
    echo "[Task 1/2] Concurrency Throughput Scan (1,2,4,8)"
    bash scripts/run_concurrency_scan.sh

    # 2.2 Prefix Caching 测试
    echo ""
    echo "[Task 2/2] Prefix Caching Benchmark"
    mkdir -p data/results/prefix_cache
    python src/benchmarks/prefix_caching.py \
        --model-path "/root/autodl-tmp/modelscope/qwen/Qwen2___5-7B-Instruct" \
        --output-dir data/results/prefix_cache

    echo ""
    echo "========================================"
    echo "Phase 2 completed!"
    echo "All results saved to data/results/"
    echo "All logs saved to logs/"
    echo "========================================"

} 2>&1 | tee -a "$LOG_FILE"
