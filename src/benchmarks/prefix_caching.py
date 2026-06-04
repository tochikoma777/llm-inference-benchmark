import sys
import os
import time
import argparse
import json
import gc

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.metrics.gpu_monitor import GPUMonitor
import torch

def cleanup_vllm():
    """彻底清理vLLM显存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    time.sleep(10)  # 给CUDA driver充分时间回收
    print("[Cleanup] GPU memory released.")

def benchmark_prefix_caching(model_path, enable_prefix_caching):
    """
    测试多轮对话场景下，Prefix Caching 的效果。
    注意：vLLM的generate()是同步的，测量的是总生成时间。
    对于短输出（128 tokens），总时间主要由预填充阶段决定，可近似反映TTFT差异。
    """
    print(f"\n{'='*60}")
    print(f"Prefix Caching: {enable_prefix_caching}")
    print(f"{'='*60}")
    
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    
    print(f"[vLLM] Loading with prefix_caching={enable_prefix_caching}...")
    
    # 降低显存占用，避免OOM
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        dtype="float16",
        enable_prefix_caching=enable_prefix_caching,
        gpu_memory_utilization=0.85,
        max_model_len=8192,  # ← 新增
    )
    
    # 3轮对话，每轮都带完整history
    # 构造 1000+ tokens 的长共享前缀（模拟长文档/系统提示）
    long_doc = (
        "请基于以下技术文档回答问题。文档内容：大语言模型（LLM）基于Transformer架构，"
        "通过自注意力机制处理序列数据。预训练阶段使用海量文本数据，通过预测下一个token"
        "来学习语言表示。微调阶段使用特定领域数据对齐模型行为。推理阶段的核心挑战包括"
        "内存带宽瓶颈、KV Cache管理、批量调度等。RAG通过外部知识库提升模型事实性。"
        "Agent架构通过工具调用扩展模型能力。多模态大模型融合视觉和语言理解。模型量化"
        "通过降低精度减少显存占用。模型蒸馏通过小模型模仿大模型行为。以上是大语言模型"
        "技术的核心要点总结。"
    ) * 10

    conversation = [
        {"role": "user", "content": long_doc + "什么是自注意力机制？"},
        {"role": "assistant", "content": ""},
        {"role": "user", "content": long_doc + "KV Cache有什么作用？"},
        {"role": "assistant", "content": ""},
        {"role": "user", "content": long_doc + "模型量化是什么？"},
    ]

    # 同时修改 sampling_params，减少 decode 时间：
    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=1)  # ← 从 128 改为 32
    
    round_times = []
    
    # 第1轮：无cache可用
    messages = [conversation[0]]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    start = time.perf_counter()
    outputs = llm.generate([text], sampling_params)
    end = time.perf_counter()
    r1_time = end - start
    round_times.append(r1_time)
    print(f"[Round 1] Total time: {r1_time:.3f}s | Input tokens: {len(outputs[0].prompt_token_ids)} (no cache possible)")
    
    # 第2轮：包含第1轮的history
    messages = [conversation[0], 
                {"role": "assistant", "content": outputs[0].outputs[0].text},
                conversation[2]]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    input_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(input_ids) > 8000:
        input_ids = input_ids[:8000]
        text = tokenizer.decode(input_ids, skip_special_tokens=False)
        print(f"[Warning] Round prompt truncated to 8000 tokens")
    
    start = time.perf_counter()
    outputs = llm.generate([text], sampling_params)
    end = time.perf_counter()
    r2_time = end - start
    round_times.append(r2_time)
    print(f"[Round 2] Total time: {r2_time:.3f}s | Input tokens: {len(outputs[0].prompt_token_ids)}")
    
    # 第3轮：包含前2轮的history
    messages = [conversation[0],
                {"role": "assistant", "content": outputs[0].outputs[0].text},
                conversation[2],
                {"role": "assistant", "content": outputs[0].outputs[0].text},
                conversation[4]]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    start = time.perf_counter()
    outputs = llm.generate([text], sampling_params)
    end = time.perf_counter()
    r3_time = end - start
    round_times.append(r3_time)
    print(f"[Round 3] Total time: {r3_time:.3f}s | Input tokens: {len(outputs[0].prompt_token_ids)}")
    
    # 显式销毁
    del llm
    cleanup_vllm()
    
    return {
        "enable_prefix_caching": enable_prefix_caching,
        "round1_time": r1_time,
        "round2_time": r2_time,
        "round3_time": r3_time,
        "speedup_round2": r1_time / r2_time if r2_time > 0 else 0,
        "speedup_round3": r1_time / r3_time if r3_time > 0 else 0,
    }

def main():
    parser = argparse.ArgumentParser(description="Phase 2: Prefix Caching Benchmark")
    parser.add_argument("--model-path", type=str, 
                        default="/root/autodl-tmp/modelscope/qwen/Qwen2___5-7B-Instruct")
    parser.add_argument("--output-dir", type=str, default="data/results/prefix_cache")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 测试关闭 prefix caching
    result_off = benchmark_prefix_caching(args.model_path, enable_prefix_caching=False)
    with open(f"{args.output_dir}/prefix_off.json", "w", encoding="utf-8") as f:
        json.dump(result_off, f, ensure_ascii=False, indent=2)
    
    print("\n[Info] Waiting 15s before next test to ensure GPU memory fully released...")
    time.sleep(15)
    
    # 测试开启 prefix caching
    result_on = benchmark_prefix_caching(args.model_path, enable_prefix_caching=True)
    with open(f"{args.output_dir}/prefix_on.json", "w", encoding="utf-8") as f:
        json.dump(result_on, f, ensure_ascii=False, indent=2)
    
    # 对比报告
    print("\n" + "="*60)
    print("Prefix Caching Comparison")
    print("="*60)
    print(f"{'Round':>8} | {'Cache OFF (s)':>14} | {'Cache ON (s)':>14} | {'Speedup':>10}")
    print("-" * 55)
    print(f"{'Round 1':>8} | {result_off['round1_time']:>14.3f} | {result_on['round1_time']:>14.3f} | {'1.00x':>10}")
    print(f"{'Round 2':>8} | {result_off['round2_time']:>14.3f} | {result_on['round2_time']:>14.3f} | {result_off['round2_time']/result_on['round2_time']:>9.2f}x")
    print(f"{'Round 3':>8} | {result_off['round3_time']:>14.3f} | {result_on['round3_time']:>14.3f} | {result_off['round3_time']/result_on['round3_time']:>9.2f}x")
    print("\n[Note] Round 1 should be similar (no cache yet). Round 2/3 should show speedup with cache ON.")
    print("[Note] Metric measures total generation time (dominated by prefill for short outputs).")

if __name__ == "__main__":
    main()
