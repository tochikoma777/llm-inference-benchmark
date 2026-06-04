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
    time.sleep(5)

def run_single(backend_name, model_path, prompt, prompt_len, output_len=128):
    monitor = GPUMonitor(interval=0.1, device_id=0)
    backend = None
    try:
        if backend_name == "naive":
            backend = NaiveBackend(model_path)
        else:
            backend = VLLMBackend(model_path, gpu_memory_utilization=0.85)
        
        _ = backend.generate("你好", max_new_tokens=20)
        time.sleep(0.5)
        cleanup()
        
        monitor.start()
        result = backend.generate(prompt, max_new_tokens=output_len)
        monitor.stop()
        
        return {
            "backend": backend_name,
            "prompt_len": prompt_len,
            "output_len": output_len,
            "peak_memory_mb": monitor.get_peak_memory_mb(),
            "avg_gpu_util_pct": monitor.get_avg_gpu_util(),
            "total_time_ms": result["total_time"] * 1000,
        }
    finally:
        if backend is not None:
            if hasattr(backend, 'llm'):
                del backend.llm
            del backend
        cleanup()

def main():
    parser = argparse.ArgumentParser(description="Phase 3: Memory vs Sequence Length")
    parser.add_argument("--backend", type=str, required=True, choices=["naive", "vllm"])
    parser.add_argument("--model-path", type=str, 
                        default="/root/autodl-tmp/modelscope/qwen/Qwen2___5-7B-Instruct")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    
    prompt_files = [
        ("data/prompts/short_prompts.jsonl", 256),
        ("data/prompts/medium_prompts.jsonl", 512),
        ("data/prompts/long_prompts.jsonl", 1024),
        ("data/prompts/very_long_prompts.jsonl", 2048),
    ]
    
    results = []
    for filepath, target_len in prompt_files:
        fullpath = os.path.join(PROJECT_ROOT, filepath)
        if not os.path.exists(fullpath):
            print(f"[Skip] {filepath} not found")
            continue
        
        with open(fullpath, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f]
        
        prompt = lines[0]["prompt"]
        actual_len = lines[0]["actual_length"]
        
        print(f"\n{'='*60}")
        print(f"Backend: {args.backend.upper()} | Prompt: {actual_len} tokens")
        print(f"{'='*60}")
        
        result = run_single(args.backend, args.model_path, prompt, actual_len)
        results.append(result)
        
        print(f"[Result] Peak memory: {result['peak_memory_mb']} MB")
        print(f"[Result] GPU util: {result['avg_gpu_util_pct']:.1f}%")
        time.sleep(3)
    
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n[Benchmark] Saved to {args.output}")

if __name__ == "__main__":
    main()