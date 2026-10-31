# 测量方法学

这份文档说明每个数字是**怎么来的**，以及在什么情况下它可以被引用。

## 1. 时间指标

### TTFT（Time To First Token）

从请求进入 `Backend.generate()` 到第一个输出 token 可用。

- **HuggingFace**：`transformers` 的 `StoppingCriteria` 在每次前向 + 采样之后被调用。
  第一次调用即「prefill 完成且首个 token 已采样」，因此
  `times[0] − t_start` 就是 TTFT。相比「max_new_tokens=1 跑一遍」的代理法，这个做法
  不会多付一次 prefill，也不会因为第二次 prefill 的显存/缓存状态不同而失真。
- **vLLM**：通过流式引擎的异步迭代器，在首次产出非空 `token_ids` 时打点。
  这包含了调度器排队时间 —— 这正是连续批处理下最需要被观察的部分。
- **static_batch**：整批共享同一个 TTFT（补齐后一次前向），符合真实静态批处理行为。

### TPOT（Time Per Output Token）

```
TPOT = (e2e − ttft) / (output_tokens − 1)
```

分母减一，因为首 token 的成本已经计入 TTFT。`output_tokens <= 1` 时 TPOT 无定义，
记录为 `null`，聚合时自动跳过。

### 端到端延迟

`e2e = 最后一个 token 可用时间 − 请求进入引擎时间`，不含客户端排队（排队单独记录为
`queue_ms`）。

## 2. 吞吐

```
throughput = Σ output_tokens / makespan
makespan   = max(finish_ms) − min(arrival_ms)
```

用 makespan 而不是「最后一个请求的 e2e」是必要的：在泊松到达下，请求是陆续进入的，
用单请求 e2e 当分母会把吞吐算高。burst 模式下两者等价。

单请求吞吐（`throughput_tok_s` 字段）也逐条记录，用于画 CDF 和散点。

## 3. 显存

采样方式：后台线程每 `monitor.interval_ms` 读一次 NVML。

- `peak_memory_mb`：整卡已用显存峰值（含权重）。
- `baseline_memory_mb`：监控启动后的第一次采样，此时引擎已加载完毕。
- `kv_cache_delta_mb = peak − baseline`：**只属于本次生成**的增量，这是跨后端可比的量。

为什么不用 `torch.cuda.memory_allocated()`：它看不到 vLLM 的自定义分配器与 CUDA graph
预留，也看不到驱动层开销。NVML 读数才是用户真正关心的「这张卡还剩多少」。

采样间隔 100 ms，因此短于 100 ms 的瞬时峰值可能被漏掉。对比趋势可靠，绝对值有轻微低估。

## 4. 负载生成

| 模式 | 到达过程 | 用途 |
|---|---|---|
| `burst` | 全部 t=0 同时到达 | 测量吞吐上限与饱和点 |
| `constant` | 固定间隔 1/rate | 稳态服务能力 |
| `poisson` | 指数分布间隔 | 最接近真实线上流量 |

每个请求由线程池中的一个 worker 执行，`max_workers` 即并发上限。超出部分在
executor 队列中等待，与真实服务的准入队列同构。

记录四个时刻：`arrival_ms`（计划到达）、`start_ms`（实际进入引擎）、
`ttft_ms`、`finish_ms`。`queue_ms = start − arrival` 直接反映准入排队。

## 5. 聚合

`aggregate_records()` 按 `(scenario, backend, serving_mode, bucket, concurrency, repeat)`
分组，计算 mean / p50 / p95 / p99。

`bucket` 由场景决定：

- 延迟扫描：每个 `(prompt_len, output_len)` 一个 bucket
- 并发扫描：整轮一个 bucket（`load`）
- 前缀缓存：每轮对话一个 bucket
- 调度：每个策略一个 bucket

聚合只做一次，图表与报告都从这份聚合结果取数，因此不会出现「图里的 p95 和表里的不一样」。

## 6. 可复现性

- 每个随机过程（`plan_arrivals`、`generate_workload`、mock 抖动）都由配置里的 `seed` 决定。
- 结果目录里保存完整配置快照与环境指纹，重跑时可直接 `llmbench run -c <saved.yaml>`。
- 模拟器在相同种子下逐位可复现。

## 7. 模拟后端是怎么标定的

`mock` 后端不是随机数发生器，而是一个解析模型：

```
aggregate_decode(b) = decode_tok_s · min(max_speedup, b^exponent)
per_request_rate    = aggregate_decode(b) / b
```

参数取自 [历史实测报告](reference-run-rtx3090.md)（RTX 3090、Qwen2.5-7B、FP16）：

| 模式 | decode 指数 | 最大加速 | KV/token |
|---|---:|---:|---:|
| continuous_batch | 0.95 | 8.0× | 0.19 MB |
| static_batch | 0.80 | 5.5× | 0.55 MB |
| per_request | 0.30 | 2.0× | 0.94 MB |

在并发 8 下分别得到 308 / 229 / 86 tok/s，与实测的 321 / 200 / — 处于同一量级。
KV 增量在 2048 token 时为 416 MB vs 实测 388 MB，节省比例 79.8% 与实测完全一致。

**它适合用来验证流水线、演示趋势、做 CI 冒烟；不适合用来引用绝对性能。**
所有模拟运行的 `metadata.system.mock = true`，报告顶部会显示警示横幅。

## 8. 什么情况下结论会失效

- 模型与硬件差异：所有标定都基于 7B / 3090，换模型或换卡请重新跑实测。
- 极短请求：TTFT 被调度抖动主导，100 ms 采样间隔不足以刻画。
- 超高并发（> 数百）：本项目用线程模拟并发，GIL 与线程切换会成为噪声来源，
  此时应改用多进程或直连 HTTP 服务端。
