import sys
import os
import time
import argparse
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.backends.naive_backend import NaiveBackend
from src.backends.vllm_backend import VLLMBackend
from src.metrics.gpu_monitor import GPUMonitor
import torch

def run_benchmark(backend_name, model_path, prompt, max_new_tokens):
    print(f"\n{'='*60}")
    print(f"Backend: {backend_name.upper()}")
    print(f"{'='*60}")
    
    monitor = GPUMonitor(interval=0.1, device_id=0)
    monitor.start()
    
    if backend_name == "naive":
        backend = NaiveBackend(model_path)
    elif backend_name == "vllm":
        backend = VLLMBackend(model_path)
    else:
        raise ValueError(f"Unknown backend: {backend_name}")
    
    print("[Benchmark] Warming up...")
    _ = backend.generate("你好", max_new_tokens=20)
    time.sleep(1)
    
    monitor.data = []
    print(f"[Benchmark] Prompt: {prompt[:50]}... | max_new_tokens={max_new_tokens}")
    
    result = backend.generate(prompt, max_new_tokens=max_new_tokens)
    
    monitor.stop()
    gpu_stats = monitor.summary()
    
    result.update({
        "backend": backend_name,
        "peak_memory_mb": gpu_stats["peak_memory_mb"],
        "avg_gpu_util": gpu_stats["avg_gpu_util"],
        "gpu_monitor_samples": gpu_stats["samples"],
    })
    
    print(f"[Result] Input tokens:  {result['num_input_tokens']}")
    print(f"[Result] Output tokens: {result['num_output_tokens']}")
    print(f"[Result] Total time:    {result['total_time']:.2f} s")
    print(f"[Result] Throughput:    {result['throughput']:.2f} tok/s")
    print(f"[Result] Peak GPU mem: {result['peak_memory_mb']} MB")
    print(f"[Result] Avg GPU util:  {result['avg_gpu_util']:.1f}%")
    print(f"[Result] Output: {result['output_text'][:100]}...")
    
    return result

def main():
    parser = argparse.ArgumentParser(description="Phase 1: Single Request Baseline")
    parser.add_argument("--backend", type=str, required=True, choices=["naive", "vllm"])
    parser.add_argument("--model-path", type=str, 
                        default="/root/autodl-tmp/modelscope/hub/Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--prompt", type=str, 
                        default="请详细解释量子计算的基本原理，以及它与传统计算机的区别。要求通俗易懂。")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    
    result = run_benchmark(args.backend, args.model_path, args.prompt, args.max_new_tokens)
    
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[Benchmark] Result saved to {args.output}")
    
    return result

if __name__ == "__main__":
    main()
