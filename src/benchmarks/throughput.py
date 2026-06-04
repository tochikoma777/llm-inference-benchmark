import sys
import os
import time
import argparse
import json
import threading
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.backends.naive_backend import NaiveBackend
from src.backends.vllm_backend import VLLMBackend
from src.metrics.gpu_monitor import GPUMonitor
import torch

def run_concurrent_naive(backend, prompts, max_new_tokens):
    """Static batching: send all prompts as one batch."""
    return backend.generate_batch(prompts, max_new_tokens=max_new_tokens)

def run_concurrent_vllm(backend, prompts, max_new_tokens):
    """vLLM: submit all prompts at once, let internal scheduler handle continuous batching."""
    return backend.generate_batch(prompts, max_new_tokens=max_new_tokens)

def cleanup_backend(backend):
    """显式清理backend，强制释放GPU显存"""
    print("[Cleanup] Destroying backend and releasing GPU memory...")
    if backend is not None:
        # 对vLLM backend，尝试显式销毁engine
        if hasattr(backend, 'llm') and hasattr(backend.llm, 'llm_engine'):
            try:
                del backend.llm
            except Exception as e:
                print(f"[Cleanup] vLLM engine cleanup warning: {e}")
        del backend
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    time.sleep(5)  # 给CUDA driver时间回收
    print("[Cleanup] Done.")

def benchmark_concurrency(backend_name, model_path, prompts, max_new_tokens, concurrency):
    print(f"\n{'='*60}")
    print(f"Backend: {backend_name.upper()} | Concurrency: {concurrency}")
    print(f"{'='*60}")
    
    monitor = GPUMonitor(interval=0.1, device_id=0)
    monitor.start()
    
    backend = None
    try:
        if backend_name == "naive":
            backend = NaiveBackend(model_path)
        else:
            # 降低gpu_memory_utilization，给后续测试留空间
            backend = VLLMBackend(model_path, gpu_memory_utilization=0.85)
        
        # 预热
        print("[Benchmark] Warming up...")
        _ = backend.generate("你好", max_new_tokens=20)
        time.sleep(1)
        
        # 构造并发请求
        test_prompts = prompts[:concurrency]
        if len(test_prompts) < concurrency:
            test_prompts = [prompts[i % len(prompts)] for i in range(concurrency)]
        
        monitor.data = []
        global_start = time.perf_counter()
        
        if backend_name == "naive":
            results = run_concurrent_naive(backend, test_prompts, max_new_tokens)
        else:
            results = run_concurrent_vllm(backend, test_prompts, max_new_tokens)
        
        global_end = time.perf_counter()
        monitor.stop()
        
        # 过滤失败请求
        valid_results = [r for r in results if r is not None]
        if not valid_results:
            print("[Error] All requests failed!")
            return None
        
        total_time = global_end - global_start
        total_output_tokens = sum(r["num_output_tokens"] for r in valid_results)
        throughput = total_output_tokens / total_time

        result = {
            "backend": backend_name,
            "concurrency": concurrency,
            "num_requests": len(valid_results),
            "total_time": total_time,
            "total_output_tokens": total_output_tokens,
            "throughput": throughput,
            "peak_memory_mb": monitor.get_peak_memory_mb(),
            "avg_gpu_util": monitor.get_avg_gpu_util(),
        }

        if backend_name == "naive":
            # Naive 是 static batching：所有请求打包成一个矩阵，同时开始同时结束
            # 每个请求的"完成时间"都是整个 batch 的总时间
            result["batch_completion_time"] = total_time
            result["avg_request_time"] = total_time  # 保留兼容旧字段
        else:
            # vLLM 是 continuous batching：每个请求独立调度，各自计时
            ttfts = [r["total_time"] for r in valid_results]
            result["avg_request_time"] = sum(ttfts) / len(ttfts)
        
        print(f"[Result] Total time:        {total_time:.2f} s")
        print(f"[Result] Total tokens:      {total_output_tokens}")
        print(f"[Result] Throughput:        {throughput:.2f} tok/s")
        print(f"[Result] Avg request time:  {result['avg_request_time']:.2f} s")
        print(f"[Result] Peak GPU mem:      {result['peak_memory_mb']} MB")
        print(f"[Result] Avg GPU util:      {result['avg_gpu_util']:.1f}%")
        
        return result
        
    finally:
        # 无论成功失败，都强制清理
        cleanup_backend(backend)

def main():
    parser = argparse.ArgumentParser(description="Phase 2: Concurrency Throughput Benchmark")
    parser.add_argument("--backend", type=str, required=True, choices=["naive", "vllm"])
    parser.add_argument("--model-path", type=str, 
                        default="/root/autodl-tmp/modelscope/qwen/Qwen2___5-7B-Instruct")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    
    prompts = [
        "什么是光合作用？简要解释。",
        "请详细解释量子计算的基本原理，以及它与传统计算机的区别。要求通俗易懂。",
        "请阅读以下背景并总结关键信息：大语言模型（LLM）基于Transformer架构，通过自注意力机制处理序列数据。预训练阶段使用海量文本数据，通过预测下一个token来学习语言表示。微调阶段使用特定领域数据对齐模型行为。推理阶段的核心挑战包括内存带宽瓶颈、KV Cache管理、批量调度等。请总结上述内容的核心要点。",
        "写一首关于春天的七言绝句。",
        "解释牛顿三大定律，并举例说明。",
        "什么是区块链？它如何保证数据不可篡改？",
        "请用简单语言解释机器学习中的过拟合和欠拟合。",
        "描述一下你最喜欢的季节，并说明原因。",
    ]
    
    result = benchmark_concurrency(
        args.backend, args.model_path, prompts, args.max_new_tokens, args.concurrency
    )
    
    if args.output and result:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[Benchmark] Result saved to {args.output}")

if __name__ == "__main__":
    main()
