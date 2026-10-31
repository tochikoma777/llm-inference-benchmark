# 架构说明

## 分层

```
┌──────────────────────────────────────────────────────────────┐
│  cli.py            run / report / dashboard / sim / …          │
├──────────────────────────────────────────────────────────────┤
│  runner/           scenarios + load generator + executive      │
├──────────────────────────────────────────────────────────────┤
│  backends/         vllm   naive(HF)   mock                     │
├──────────────────────────────────────────────────────────────┤
│  core/             config · paths · logging · results          │
│  metrics/          NVML telemetry · statistics                 │
├──────────────────────────────────────────────────────────────┤
│  reporting/        analysis · html · markdown                  │
│  viz/              theme · plotly · matplotlib                 │
│  dashboard/        streamlit app                               │
└──────────────────────────────────────────────────────────────┘
```

依赖方向自上而下，没有任何反向依赖：`viz` 不认识 `runner`，`backends` 不认识 `viz`。
唯一的例外是 `reporting/analysis` 依赖 `core/results`（它本来就是结果的一层视图）。

## 关键设计决策

### 1. 后端负责计时，运行器负责编排

`Backend.generate()` 返回的 `GenerationOutput` 里已经带有 `ttft_ms` / `e2e_ms` /
`token_times_ms`。运行器不猜、不推算、不做「用 1 token 的耗时当 TTFT」这类近似。

原因：只有引擎内部知道「第一个 token 可用」的准确时刻。把它交给运行器，势必要
在外部轮询，既不准也不通用。

### 2. 场景只产出原子记录，聚合只有一处

场景把每条请求（或每个网格点）写成一行记录，`aggregate_records()` 负责所有统计。
好处：任何一张图都能直接对应到一组原始记录；新增图表不需要重新计算指标；
不同报告之间不会出现口径漂移。

### 3. 结果落盘是 JSONL + metadata + summary

- `results.jsonl`：追加友好、可逐行排查、可被任何语言读取。
- `metadata.json`：环境指纹、配置快照、运行状态。没有它，一个数字就没有意义。
- `summary.json`：预聚合结果，供快速查看与外部工具消费。

### 4. 模拟后端是一等公民

`mock` 不是测试替身，而是完整实现 `Backend` 契约的第五个后端。它让：

- 新贡献者 5 分钟内看到完整产物
- CI 可以端到端验证报告生成
- 文档截图可以随时重生成

代价是必须严防「模拟数据被当成实测」，因此 `metadata.system.mock` 与
`is_mock` 字段贯穿到报告与仪表盘的 UI。

### 5. 可视化与分析分离

`reporting/analysis.py` 输出 **DataFrame**，`viz/plotly_charts.py` 与
`viz/figures.py` 只负责把 DataFrame 画出来，`dashboard/app.py` 复用同一批
Plotly 构建函数。新增一个前端不需要重写任何指标逻辑。

### 6. 配置先校验再运行

`RunConfig.validate()` 在创建任何引擎之前执行。典型的失败模式（并发为 0、
未知后端、显存利用率越界）会在 0.1 秒内报错，而不是在加载 14 GB 权重之后。

## 调度模拟器

`Simulation.run()` 是一个迭代级循环，每一步对应 vLLM 的一个 scheduler step：

```
1. 收取到达请求 → waiting
2. 按策略排序 waiting，在 max_num_seqs 与块池允许的范围内准入
3. 计算本步耗时：prefill_tokens/prefill_rate + decode_step·(1+(b−1)·factor)
4. 推进时间，被准入的请求完成 prefill（并产出第一个 token），其余各产出 1 个 token
5. 完成的请求释放块；扩容失败的请求自我抢占并回到 waiting
```

与真实引擎对齐的几个细节：

- **块级按需分配**：不预分配 `prompt_len + output_len`，这正是 PagedAttention 相对
  静态分片的本质改进。
- **不抢占入场**：等待队列中的请求不会抢占正在运行的请求（vLLM 默认行为），
  这消除了「两个请求互相抢占」的活锁。
- **抢占即重算**：被抢占的请求释放全部块，`computed_tokens` 归零，之后重新 prefill；
  已产出的 token 不重复计数。
- **不可行请求丢弃**：若一个请求的 KV 需求超过整个池容量，或反复抢占超过 8 次，
  直接丢弃并计入 `dropped`，避免模拟不收敛。

`Simulation` 是纯函数式的：给定相同 workload 与 seed，输出逐位一致。政策对比时每个
策略都会拿到 workload 的深拷贝，互不干扰。

## 性能考量

- 引擎懒加载：`import llmbench` 不会 import torch。
- 结果加载用 pandas 一次性读入，聚合用纯 Python 实现，万级记录毫秒级完成。
- 图表渲染可关闭（`--no-figures`），CI 里只跑 HTML 时省去 matplotlib 的开销。
- 所有遥测采样在守护线程中进行，不阻塞基准循环。

## 扩展点

| 想扩展 | 改哪里 |
|---|---|
| 新推理引擎 | `backends/` + `register_backend` |
| 新场景 | `runner/scenarios.py` + `executive.SCENARIOS` |
| 新指标 | `core/results.aggregate_records` + `reporting/analysis` |
| 新图表 | `viz/plotly_charts.py`（交互）或 `viz/figures.py`（静态） |
| 新报告格式 | `reporting/`，输入统一是 `Analysis` |
| 新调度策略 | `scheduler_sim/policies.py` + `POLICIES` |
