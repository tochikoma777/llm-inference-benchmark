# LLM Inference Benchmark — 自动生成报告

> ⚠️ **模拟数据**：本报告由 mock 后端生成，仅用于验证流水线，不可作为硬件实测结论。

- **运行**: `20261031-130129-sim-per-request-0efec4`
- **后端**: `mock` (per_request)
- **模型**: `—`
- **GPU**: NVIDIA RTX 3090 24GB (simulated)
- **时间**: 2026-10-31T13:01:29.024876+00:00


## 关键指标

| 指标 | 数值 | 说明 |
|:---|:---|:---|
| 峰值吞吐 | **308 tok/s** | mock · continuous_batch @ c=8.0 |
| 吞吐加速比 | **3.60×** | mock · continuous_batch vs mock · per_request |
| 最优 TTFT p95 | **82.8ms** | 高并发下 |
| KV Cache 节省 | **80%** | mock · continuous_batch @ 2.1k tokens |
| 前缀缓存加速 | **8.66×** | 多轮对话末轮 TTFT |
| 最优调度策略 | **SJF** | 平均 TTFT 4.09s |

## 后端对比（最高并发）

| Series | Serving mode | Concurrency | Throughput (tok/s) | TTFT mean (ms) | TTFT p95 (ms) | E2E p99 (ms) | GPU util (%) | Peak memory (MB) | Speed-up |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| mock · continuous_batch | continuous_batch | 8 | 308.34 | 91.30 | 206.42 | 3316.24 | 84.70 | 14448.14 | 3.60 |
| mock · per_request | per_request | 8 | 85.63 | 177.25 | 385.26 | 11909.16 | 52.27 | 15427.64 | 1 |
| mock · static_batch | static_batch | 8 | 228.80 | 251.65 | 529.09 | 4464.17 | 72.50 | 14918.30 | 2.67 |


## 并发扩展

| 后端 | 并发 | 吞吐 (tok/s) | TTFT p95 (ms) | E2E p99 (ms) | GPU 利用率 (%) |
|:---|:---|:---|:---|:---|:---|
| mock · continuous_batch | 1 | 42.63 | 97.68 | 3002.03 | 70.75 |
| mock · continuous_batch | 2 | 85.66 | 82.78 | 2987.89 | 75.08 |
| mock · continuous_batch | 4 | 152.28 | 408.05 | 3355.96 | 80.28 |
| mock · continuous_batch | 8 | 308.34 | 206.42 | 3316.24 | 84.70 |
| mock · per_request | 1 | 42.63 | 97.68 | 3002.03 | 64.50 |
| mock · per_request | 2 | 55.66 | 83.98 | 4582.38 | 62.25 |
| mock · per_request | 4 | 66.36 | 426.98 | 7674.19 | 57.75 |
| mock · per_request | 8 | 85.63 | 385.26 | 11909.16 | 52.27 |
| mock · static_batch | 1 | 42.63 | 97.68 | 3002.03 | 83 |
| mock · static_batch | 2 | 77.76 | 97.32 | 3288.33 | 81.50 |
| mock · static_batch | 4 | 125.21 | 442.67 | 4077.31 | 78.50 |
| mock · static_batch | 8 | 228.80 | 529.09 | 4464.17 | 72.50 |


## 前缀缓存（多轮对话）

| 分组 | 输入长度 | TTFT (ms) | 端到端 (ms) |
|:---|:---|:---|:---|
| cache_off_r1 | 1,376 | 468.33 | 2646.58 |
| cache_off_r2 | 1,460 | 475.63 | 2560.61 |
| cache_off_r3 | 1,541 | 527.78 | 2719.72 |
| cache_on_r1 | 1,376 | 468.33 | 2646.58 |
| cache_on_r2 | 1,460 | 31.60 | 2116.57 |
| cache_on_r3 | 1,541 | 60.96 | 2252.90 |


## 调度策略对比

| 策略 | 平均 TTFT (ms) | TTFT p95 (ms) | E2E p99 (ms) | 吞吐 (tok/s) | 平均排队 (ms) | 抢占次数 | 公平性 (Jain) |
|:---|:---|:---|:---|:---|:---|:---|:---|
| FCFS | 6934.64 | 25294.71 | 62192.12 | 188.44 | 6643.31 | 0 | 0.37 |
| SJF + aging | 4191.67 | 21214.86 | 65043.83 | 190.40 | 3908.31 | 0 | 0.45 |
| Priority | 4314.22 | 23363.01 | 62602.97 | 189.19 | 4028.27 | 0 | 0.45 |
| SJF (oracle) | 4089.78 | 20373.51 | 72645.51 | 190.32 | 3803.19 | 0 | 0.46 |


## 复现

```bash
llmbench run -c config/vllm.yaml --model-path /path/to/model
llmbench report --latest
```
