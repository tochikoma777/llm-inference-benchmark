#!/usr/bin/env python3
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoTokenizer

TOPICS = [
    "请详细解释量子计算的基本原理，以及它与传统计算机的区别。量子比特利用叠加态和纠缠态实现并行计算，在特定问题上具有指数级加速潜力。",
    "大语言模型基于Transformer架构，通过自注意力机制处理序列数据。预训练阶段使用海量文本，通过预测下一个token学习语言表示。微调阶段使用特定领域数据对齐模型行为。",
    "机器学习中的梯度下降算法通过反向传播更新参数，损失函数衡量预测与真实值的差距，学习率控制更新步长。优化器如Adam自适应调整学习率。",
    "区块链技术的核心机制包括分布式账本、共识算法和加密哈希。比特币使用工作量证明，以太坊转向权益证明。智能合约在链上自动执行。",
    "神经网络模拟生物神经元连接，多层感知机通过非线性激活函数拟合复杂函数。卷积神经网络擅长图像处理，循环神经网络处理序列数据。",
    "操作系统负责进程调度、内存管理和文件系统。Linux采用宏内核架构，微内核将驱动移到用户态。虚拟内存通过页表实现地址转换。",
    "数据库索引通过B+树或哈希表加速查询。事务遵循ACID原则，分布式数据库引入CAP定理权衡一致性、可用性和分区容错。",
    "编译器将高级语言转换为机器码，经历词法分析、语法分析、语义分析和代码优化阶段。LLVM采用中间表示实现跨平台优化。",
]

def generate_prompt(tokenizer, target_len, topic_idx):
    text = ""
    loop = 0
    while len(tokenizer.encode(text, add_special_tokens=False)) < target_len:
        text += TOPICS[(topic_idx + loop) % len(TOPICS)] + "\n\n"
        loop += 1
    
    tokens = tokenizer.encode(text, add_special_tokens=False)[:target_len]
    prompt = tokenizer.decode(tokens, skip_special_tokens=False)
    actual_len = len(tokenizer.encode(prompt, add_special_tokens=False))
    return prompt, actual_len

def main():
    model_path = "/root/autodl-tmp/modelscope/qwen/Qwen2___5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    )
    
    config = [
        ("short", 256, 100),
        ("medium", 512, 50),
        ("long", 1024, 50),
        ("very_long", 2048, 20),
    ]
    
    output_dir = os.path.join(PROJECT_ROOT, "data", "prompts")
    os.makedirs(output_dir, exist_ok=True)
    
    for name, target_len, count in config:
        prompts = []
        for i in range(count):
            prompt, actual_len = generate_prompt(tokenizer, target_len, i)
            prompts.append({
                "prompt": prompt,
                "target_length": target_len,
                "actual_length": actual_len,
            })
        
        filepath = os.path.join(output_dir, f"{name}_prompts.jsonl")
        with open(filepath, "w", encoding="utf-8") as f:
            for p in prompts:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        
        lengths = [p["actual_length"] for p in prompts]
        print(f"[{name}] Generated {count} prompts | target={target_len} "
              f"actual_avg={sum(lengths)/len(lengths):.0f} min={min(lengths)} max={max(lengths)}")
        print(f"  Saved to {filepath}")

if __name__ == "__main__":
    main()