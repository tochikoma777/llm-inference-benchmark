# LLM Inference Benchmark

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![vLLM 0.5.5](https://img.shields.io/badge/vLLM-0.5.5-green.svg)](https://github.com/vllm-project/vllm)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

系统性评估 vLLM vs HuggingFace Transformers 在大模型推理中的性能差异，
覆盖 Continuous Batching、Prefix Caching、KV Cache 管理、调度策略四大核心机制。

## 核心发现

| 场景 | Naive (HF) | vLLM | 提升 |
|------|-----------|------|------|
| 单请求吞吐 | 30.77 tok/s | 48.39 tok/s | 1.57x |
| 8并发吞吐 | 199.71 tok/s | 321.22 tok/s | 1.61x |
| KV Cache显存(2048) | +1922 MB | +388 MB | 节省4.9x |
| GPU利用率(8并发) | 61.8% | 89.0% | - |

## 快速开始

```bash
# 1. 环境准备
bash env/setup.sh

# 2. 下载模型（ModelScope）
python scripts/download_model.py

# 3. 运行全部测试
bash scripts/run_all_benchmarks.sh
```

## 项目结构

```bash
llm-inference-benchmark/
├── src/
│   ├── backends/
│   │   ├── naive_backend.py      # HF Transformers 静态batching
│   │   └── vllm_backend.py       # vLLM 连续batching
│   ├── benchmarks/
│   │   ├── single_request.py     # 阶段1：单请求基线
│   │   ├── throughput.py         # 阶段2：并发吞吐
│   │   ├── prefix_caching.py     # 阶段2：前缀缓存
│   │   ├── latency_breakdown.py  # 阶段3：TTFT/TPOT拆解
│   │   └── memory_scan.py        # 阶段3：显存增长扫描
│   ├── metrics/
│   │   └── gpu_monitor.py        # GPU显存/利用率监控
│   ├── scheduler_sim/
│   │   ├── base_scheduler.py     # 调度器基类
│   │   ├── fcfs_scheduler.py     # 先来先服务
│   │   ├── sjf_scheduler.py      # 短作业优先
│   │   └── benchmark.py          # 调度策略对比
│   └── utils/
│       └── report_generator.py   # 图表生成
├── data/
│   ├── prompts/                  # 测试数据集（256/512/1024/2048 tokens）
│   └── results/                  # 实验结果（JSON）
├── reports/
│   ├── figures/                  # 分析图表
│   └── benchmark_report.md       # 技术报告
├── scripts/
│   ├── run_baseline.sh           # 阶段1一键运行
│   ├── run_phase2.sh             # 阶段2一键运行
│   ├── run_phase3.sh             # 阶段3一键运行
│   └── run_scheduler.sh          # 调度模拟器
└── env/
└── requirements.txt
```

## 技术栈

- 模型：Qwen2.5-7B-Instruct
- 推理框架：vLLM 0.5.5 / HuggingFace Transformers 4.44.2
- 硬件：RTX 3090 24GB / AutoDL
- 数据分析：pandas + matplotlib

## 相关资源

- 技术报告：[reports/benchmark_report.md](待完成)
- 技术博客：[从RTL到vLLM：一个芯片工程师的云端大模型推理优化实践](待完成)

## License

MIT