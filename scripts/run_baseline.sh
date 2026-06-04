#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# 日志配置
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/phase1_baseline_${TIMESTAMP}.log"

MODEL_PATH="/root/autodl-tmp/modelscope/qwen/Qwen2___5-7B-Instruct"
PROMPT="请详细解释量子计算的基本原理，以及它与传统计算机的区别。要求通俗易懂。"
MAX_TOKENS=256

mkdir -p data/results

# 使用 {} 包裹 + 管道 tee，替代 exec 进程替换
{
    echo "========================================"
    echo "Phase 1: Single Request Baseline"
    echo "Model: Qwen2.5-7B-Instruct"
    echo "Local path: $MODEL_PATH"
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo "Log file: $LOG_FILE"
    echo "========================================"

    # 验证路径存在
    if [ ! -d "$MODEL_PATH" ]; then
        echo "[Error] Model path not found: $MODEL_PATH"
        exit 1
    fi

    echo "[Info] Model files found: $(ls $MODEL_PATH | wc -l) files"

    # 1. Naive Backend
    echo ""
    echo "[Step 1/2] Running Naive Backend (HuggingFace)..."
    python src/benchmarks/single_request.py \
        --backend naive \
        --model-path "$MODEL_PATH" \
        --prompt "$PROMPT" \
        --max-new-tokens "$MAX_TOKENS" \
        --output data/results/naive_baseline.json

    echo ""
    echo "[Info] Naive completed. Waiting 5s for GPU memory cleanup..."
    sleep 5

    # 2. vLLM Backend
    echo ""
    echo "[Step 2/2] Running vLLM Backend..."
    python src/benchmarks/single_request.py \
        --backend vllm \
        --model-path "$MODEL_PATH" \
        --prompt "$PROMPT" \
        --max-new-tokens "$MAX_TOKENS" \
        --output data/results/vllm_baseline.json

    echo ""
    echo "========================================"
    echo "Baseline tests completed!"
    echo "Log saved to: $LOG_FILE"
    echo "========================================"

    echo ""
    echo "[Summary] Loading results..."
    python3 << 'PYEOF'
import json
with open("data/results/naive_baseline.json") as f:
    n = json.load(f)
with open("data/results/vllm_baseline.json") as f:
    v = json.load(f)

print(f"{'Metric':<<20} {'Naive (HF)':>15} {'vLLM':>15} {'Speedup':>10}")
print("-" * 65)
print(f"{'Total Time (s)':<<20} {n['total_time']:>15.2f} {v['total_time']:>15.2f} {n['total_time']/v['total_time']:>9.2f}x")
print(f"{'Throughput (tok/s)':<<20} {n['throughput']:>15.2f} {v['throughput']:>15.2f} {v['throughput']/n['throughput']:>9.2f}x")
print(f"{'Peak Memory (MB)':<<20} {n['peak_memory_mb']:>15} {v['peak_memory_mb']:>15} {'-':>10}")
print(f"{'Avg GPU Util (%)':<<20} {n['avg_gpu_util']:>15.1f} {v['avg_gpu_util']:>15.1f} {'-':>10}")
PYEOF

    echo ""
    echo "Results saved to data/results/"

} 2>&1 | tee -a "$LOG_FILE"
