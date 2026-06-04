#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/phase3_${TIMESTAMP}.log"

{
    echo "========================================"
    echo "Phase 3: Systematic Performance Evaluation"
    echo "Start time: $(date)"
    echo "Log file: $LOG_FILE"
    echo "========================================"

    # Step 1: Generate prompts
    echo ""
    echo "[Step 1/5] Generating test datasets..."
    python scripts/generate_prompts.py

    # Step 2: TTFT/TPOT latency breakdown
    echo ""
    echo "[Step 2/5] TTFT/TPOT latency breakdown..."
    mkdir -p data/results/latency
    
    for BACKEND in naive vllm; do
        for NAME in short medium long very_long; do
            FILE="data/prompts/${NAME}_prompts.jsonl"
            if [ -f "$FILE" ]; then
                echo "[Latency] backend=$BACKEND prompt=$NAME"
                python src/benchmarks/latency_breakdown.py \
                    --backend $BACKEND \
                    --prompt-file $FILE \
                    --output-lens 1 128 256 \
                    --output "data/results/latency/${BACKEND}_${NAME}.json" \
                    || echo "[WARN] $BACKEND $NAME failed"
                sleep 5
            fi
        done
    done

    # Step 3: Memory scan
    echo ""
    echo "[Step 3/5] Memory scan vs sequence length..."
    mkdir -p data/results/memory
    for BACKEND in naive vllm; do
        echo "[Memory] backend=$BACKEND"
        python src/benchmarks/memory_scan.py \
            --backend $BACKEND \
            --output "data/results/memory/${BACKEND}_memory.json" \
            || echo "[WARN] $BACKEND memory scan failed"
        sleep 5
    done

    # Step 4: Generate figures
    echo ""
    echo "[Step 4/5] Generating analysis figures..."
    python src/utils/report_generator.py

    echo ""
    echo "========================================"
    echo "Phase 3 completed!"
    echo "End time: $(date)"
    echo "Figures: reports/figures/"
    echo "Data:    data/results/"
    echo "========================================"
} 2>&1 | tee -a "$LOG_FILE"