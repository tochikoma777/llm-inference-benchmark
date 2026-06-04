#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/phase2_concurrency_${TIMESTAMP}.log"

MODEL_PATH="/root/autodl-tmp/modelscope/qwen/Qwen2___5-7B-Instruct"
MAX_TOKENS=128

mkdir -p data/results/concurrency

{
    echo "========================================"
    echo "Phase 2: Concurrency Throughput Scan"
    echo "Model: Qwen2.5-7B-Instruct"
    echo "Testing concurrency: 1, 2, 4, 8"
    echo "Log file: $LOG_FILE"
    echo "========================================"

    for CONC in 1 2 4 8; do
        echo ""
        echo "[Step] Testing concurrency=$CONC with Naive backend..."
        python src/benchmarks/throughput.py \
            --backend naive \
            --model-path "$MODEL_PATH" \
            --concurrency "$CONC" \
            --max-new-tokens "$MAX_TOKENS" \
            --output "data/results/concurrency/naive_c${CONC}.json" || echo "[WARN] Naive c=$CONC failed"
        
        echo "[Info] Naive c=$CONC done. GPU memory check:"
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
        sleep 10
        
        echo "[Step] Testing concurrency=$CONC with vLLM backend..."
        python src/benchmarks/throughput.py \
            --backend vllm \
            --model-path "$MODEL_PATH" \
            --concurrency "$CONC" \
            --max-new-tokens "$MAX_TOKENS" \
            --output "data/results/concurrency/vllm_c${CONC}.json" || echo "[WARN] vLLM c=$CONC failed"
        
        echo "[Info] vLLM c=$CONC done. GPU memory check:"
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
        sleep 15
    done

    echo ""
    echo "========================================"
    echo "Concurrency scan completed!"
    echo "Log saved to: $LOG_FILE"
    echo "========================================"

    echo ""
    echo "[Summary] Generating comparison table..."
    python3 << 'PYEOF'
import json
import os

results_dir = "data/results/concurrency"
concurrencies = [1, 2, 4, 8]

print(f"\n{'Concurrency':>12} | {'Backend':>8} | {'Throughput (tok/s)':>18} | {'Avg Time (s)':>14} | {'GPU Util (%)':>12}")
print("-" * 75)

for c in concurrencies:
    for backend in ["naive", "vllm"]:
        path = f"{results_dir}/{backend}_c{c}.json"
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            print(f"{c:>12} | {backend:>8} | {data['throughput']:>18.2f} | {data['avg_request_time']:>14.2f} | {data['avg_gpu_util']:>12.1f}")
        else:
            print(f"{c:>12} | {backend:>8} | {'FAILED':>18} | {'FAILED':>14} | {'FAILED':>12}")

print("\n[Interpretation]")
print("- Naive backend: 并发增加时吞吐应下降（GIL竞争），GPU利用率降低")
print("- vLLM backend: 并发增加时吞吐应接近线性增长，GPU利用率维持高位")
print("- 当 GPU 利用率接近 100% 时，吞吐增长会饱和")
PYEOF

} 2>&1 | tee -a "$LOG_FILE"
