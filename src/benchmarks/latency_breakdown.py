#!/usr/bin/env python3
import sys
import os
import time
import argparse
import json
import gc

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.backends.naive_backend import NaiveBackend
from src.backends.vllm_backend import VLLMBackend
from src.metrics.gpu_monitor import GPUMonitor
import torch

def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    time.sleep(3)

def main():
    parser = argparse.ArgumentParser(description="Phase 3: TTFT/TPOT Latency Breakdown")
    parser.add_argument("--backend", type=str, required=True, choices=["naive", "vllm"])
    parser.add_argument("--model-path", type=str, 
                        default="/root/autodl-tmp/modelscope/qwen/Qwen2___5-7B-Instruct")
    parser.add_argument("--prompt-file", type=str, required=True)
    parser.add_argument("--output-lens", type=int, nargs="+", default=[1, 128, 256])
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    
    with open(args.prompt_file, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]
    
    prompt_data = lines[0]
    prompt = prompt_data["prompt"]
    prompt_len = prompt_data["actual_length"]
    
    backend = None
    results = []
    
    try:
        print(f"[Init] Loading {args.backend.upper()} backend...")
        if args.backend == "naive":
            backend = NaiveBackend(args.model_path)
        else:
            backend = VLLMBackend(args.model_path, gpu_memory_utilization=0.85)
        
        # Warmup
        print("[Warmup] Single token generation...")
        _ = backend.generate("你好", max_new_tokens=20)
        time.sleep(0.5)
        cleanup()
        
        for out_len in args.output_lens:
            print(f"\n{'='*60}")
            print(f"Test: prompt={prompt_len} | output={out_len} | backend={args.backend.upper()}")
            print(f"{'='*60}")
            
            monitor = GPUMonitor(interval=0.1, device_id=0)
            monitor.start()
            
            result = backend.generate(prompt, max_new_tokens=out_len)
            
            monitor.stop()
            
            record = {
                "backend": args.backend,
                "prompt_len": prompt_len,
                "output_len": out_len,
                "num_input_tokens": result["num_input_tokens"],
                "num_output_tokens": result["num_output_tokens"],
                "total_time_ms": result["total_time"] * 1000,
                "throughput_tok_s": result["throughput"],
                "peak_memory_mb": monitor.get_peak_memory_mb(),
                "avg_gpu_util_pct": monitor.get_avg_gpu_util(),
            }
            results.append(record)
            
            print(f"[Result] Total time: {record['total_time_ms']:.0f} ms")
            print(f"[Result] Throughput: {record['throughput_tok_s']:.2f} tok/s")
            print(f"[Result] Peak memory: {record['peak_memory_mb']} MB")
            
            if out_len == 1:
                print(f"[Note] output_len=1 → total_time approximates TTFT")
            
            time.sleep(2)
            cleanup()
    
    finally:
        if backend is not None:
            if hasattr(backend, 'llm'):
                del backend.llm
            del backend
        cleanup()
    
    # Derive TTFT/TPOT
    ttft_ms = None
    for r in results:
        if r["output_len"] == 1:
            ttft_ms = r["total_time_ms"]
            r["ttft_ms"] = ttft_ms
    
    if ttft_ms:
        for r in results:
            if r["output_len"] > 1 and r["num_output_tokens"] > 1:
                r["ttft_ms"] = ttft_ms
                r["tpot_ms"] = (r["total_time_ms"] - ttft_ms) / (r["num_output_tokens"] - 1)
                print(f"\n[Derived] output_len={r['output_len']}: "
                      f"TTFT={ttft_ms:.0f} ms | TPOT={r['tpot_ms']:.1f} ms/token")
    
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n[Benchmark] Saved to {args.output}")

if __name__ == "__main__":
    main()