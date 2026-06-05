# 🚀 LLM Inference Benchmark

> 系统性评估 vLLM 与 HuggingFace Transformers 在 Qwen2.5-7B 上的推理性能，覆盖 Continuous Batching、Prefix Caching、KV Cache 管理、调度策略四大核心机制，输出 30+ 组实验数据与 7 张 publication-quality 图表。

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![vLLM 0.5.5](https://img.shields.io/badge/vLLM-0.5.5-green.svg)](https://github.com/vllm-project/vllm)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ 核心发现

| 场景 | Naive (HF) | vLLM | 提升 |
|------|-----------|------|------|
| 单请求吞吐 | 39.07 tok/s | 47.31 tok/s | **1.21×** |
| 8 并发吞吐 | 199.71 tok/s | 321.22 tok/s | **1.61×** |
| 8 并发尾延迟 | 5.13 s | 2.88 s | **减少 44%** |
| KV Cache 显存 (2048 tokens) | +1922 MB | +388 MB | **节省 79.8%** |
| GPU 利用率 (8 并发) | 61.8% | 89.0% | **稳定高负载** |
| Prefix Caching 加速 (TTFT) | - | **1.75×** | 多轮对话收益 |

> **一句话总结**：vLLM 通过 Continuous Batching 和 PagedAttention，将 3090 上的 7B 模型推理吞吐提升 1.6 倍，同时显存占用减少 80%，GPU 利用率稳定在 90% 附近。

---

## 🎯 项目动机

大模型从“训练时代”迈入“推理时代”。未经优化的 naive 推理 GPU 利用率不足 20%，无法满足生产环境高并发需求。vLLM 作为工业界主流推理框架，通过 PagedAttention 和 Continuous Batching 彻底改变了推理效率。本项目从一名芯片设计工程师的视角，**手把手复现并量化这些优化技术，建立起从硬件到算法的全栈理解**。

---

## 📊 实验矩阵

我们设计了 5 维度 benchmark 矩阵，覆盖 **30+ 组** 对比实验：

- **模型**：Qwen2.5-7B-Instruct  
- **硬件**：NVIDIA RTX 3090 24GB (AutoDL 云平台)  
- **序列长度**：256 / 512 / 1024 / 2048 tokens  
- **并发度**：1 / 2 / 4 / 8  
- **后端**：HuggingFace Transformers (Static Batching) vs vLLM (Continuous Batching)  
- **优化**：Prefix Caching 开/关, FCFS vs SJF 调度模拟  

所有原始数据均已开源（`data/results/`），图表见 `reports/figures/`。

---

## 📈 关键图表一览

| 图表 | 说明 |
|------|------|
| `01_ttft_vs_prompt_length.png` | 首 Token 延迟随序列长度线性增长，vLLM 略优 |
| `02_throughput_vs_concurrency.png` | vLLM 6.8× 线性扩展，Naive 5.1× 亚线性 |
| `03_memory_vs_sequence_length.png` | PagedAttention 显存优势：2048 tokens 时少用 1.5GB |
| `04_gpu_utilization.png` | vLLM 保持 90% 利用率，Naive 从 79% 降至 62% |
| `05_prefix_cache.png` | Prefix Caching 多轮对话加速 1.75× |
| `06_batching_comparison.png` | Static vs Continuous Batching 完成时间对比 |
| `07_scheduler_comparison.png` | 自研调度模拟器：SJF 短请求 TTFT 仅 0.1s，但长请求尾延迟恶化 |

---

## 🏗️ 项目结构

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


---

## 🚀 快速开始

### 1. 环境准备
```bash
bash env/setup.sh
```
自动创建 Conda 环境，安装 PyTorch 2.4.0、vLLM 0.5.5、Transformers 等依赖。

### 2. 下载模型
```bash
python scripts/download_model.py
```
使用 ModelScope 下载 Qwen2.5-7B-Instruct 至本地路径。

### 3. 运行全部实验
```bash
bash scripts/run_all_benchmarks.sh
```
一键执行所有阶段测试（约需 40 分钟，建议在 GPU 实例上运行）。

### 4. 查看结果
- 图表：reports/figures/
- 原始数据：data/results/
- 技术报告：reports/benchmark_report.md

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