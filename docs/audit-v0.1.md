# v0.1 项目审计：问题、缺陷与优化点

> **审计对象**：`tochikoma777/llm-inference-benchmark` @ `7b047a2`（已打标签 `v0.1.0`）
> **审计时间**：2026-10-31
> **方法**：逐文件阅读 + 与原报告数值交叉核对 + 对关键结论做可执行验证（`v0.2` 的 mock 后端在相同假设下能否复现原报告量级）
> **配套文档**：[决策记录](decisions.md) · [后续路线图](roadmap.md)

---

## 0. 阅读说明

### 0.1 严重级别

| 级别 | 定义 | 处理策略 |
|---|---|---|
| **P0** | 结论错误或可能得出相反结论，直接影响项目可信度 | v0.2 必须修复，否则项目失去意义 |
| **P1** | 结果可用但口径不严谨，或存在阻塞性缺陷 | v0.2 修复 |
| **P2** | 工程质量、可维护性、可复现性问题 | v0.2 修复，或列入路线图 |
| **P3** | 体验/文档细节 | 择机 |

### 0.2 如何复现审计现场

```bash
git checkout v0.1.0
git show v0.1.0:src/scheduler_sim/base_scheduler.py | nl -ba | sed -n '135,150p'
```

下文的 `文件:行号` 均指 `v0.1.0` 中的位置。

### 0.3 一句话结论

v0.1 的实验**设计意图是对的**（覆盖单请求、并发、显存、前缀缓存、调度五个维度），
但**测量实现存在系统性偏差**：TTFT 是代理指标、并发测试没有真正的并发、
调度模拟器有会改变结论的 bug。因此原报告中的具体倍数（1.61×、79.8%、1.75×）
**方向可信、精度不可引用**。

---

## 1. 结论摘要

| ID | 级别 | 问题 | 对结论的影响 |
|---|---|---|---|
| A-01 | P0 | TTFT 用「生成 1 token 的总耗时」代理 | TTFT 偏高，且污染 TPOT 推导 |
| A-02 | P0 | 并发测试没有到达过程，只是一次 batched call | 「并发扩展曲线」名不副实 |
| A-03 | P0 | 调度模拟器 p99 索引越界 / 时间片重复计数 | 调度对比结论不可靠 |
| A-04 | P0 | 显存报整卡总量而非生成增量 | 跨后端对比不公平 |
| A-05 | P1 | vLLM batch 的 per-request throughput 语义错误 | 单请求指标失真 |
| A-06 | P1 | HF 批处理把 padding 计入输出 token | 吞吐虚高 |
| A-07 | P1 | 调度器预分配 `prompt_len + output_len` | 模拟结果偏离 PagedAttention 真实行为 |
| A-08 | P1 | 分配失败即死循环 | 极端配置下模拟不收敛 |
| A-09 | P2 | 路径硬编码且三处拼写不一致 | 换机器即失败 |
| A-10 | P2 | 依赖/许可证/测试/CI 全缺 | 无法复现、无法协作 |
| A-11 | P2 | 图表依赖代理指标与硬编码路径 | 图表与数据可能脱节 |
| A-12 | P2 | README 声称开源原始数据，实际被 .gitignore | 承诺与事实不符 |

---

## 2. 测量正确性问题（P0）

### A-01 TTFT 是用「1 token 总耗时」代理出来的

**证据**：`src/benchmarks/latency_breakdown.py:99-112`

```python
ttft_ms = None
for r in results:
    if r["output_len"] == 1:
        ttft_ms = r["total_time_ms"]          # 用一次独立的 1-token 请求当 TTFT
        r["ttft_ms"] = ttft_ms
...
r["tpot_ms"] = (r["total_time_ms"] - ttft_ms) / (r["num_output_tokens"] - 1)
```

**为什么错**：TTFT 必须来自**同一个请求**。这里换了一个请求测「生成 1 个 token 要多久」，
两者之间隔着一次完整的模型加载、一次 CUDA 上下文切换与不同的 KV Cache 状态；
而且 1-token 请求本身仍然包含一次完整的 prefill，本质上是「e2e(1 token)」而非 TTFT。

**后果**：TTFT 系统性偏高；用这个 TTFT 反推 TPOT 会把误差全部推给解码阶段。

**v0.2 处理**：后端内部计时（见 [D-02](decisions.md#d-02-把计时职责下沉到后端)）。
HF 用 `StoppingCriteria` 在每步解码回调打点，vLLM 用流式迭代器在首次产出 token 时打点。

---

### A-02 「并发测试」没有并发

**证据**：`src/benchmarks/throughput.py:18-24`

```python
def run_concurrent_naive(backend, prompts, max_new_tokens):
    """Static batching: send all prompts as one batch."""
    return backend.generate_batch(prompts, max_new_tokens=max_new_tokens)

def run_concurrent_vllm(backend, prompts, max_new_tokens):
    """vLLM: submit all prompts at once, let internal scheduler handle ..."""
    return backend.generate_batch(prompts, max_new_tokens=max_new_tokens)
```

这两个函数体**完全一样**。所谓 `--concurrency 8` 只是把 prompt 列表切成 8 条一次性提交，
没有任何到达过程、没有排队、没有请求在时间轴上的重叠。

**后果**：

- 画出来的「吞吐 vs 并发」实际是「吞吐 vs 批大小」；
- 无法观测**准入排队**（这是连续批处理最关键的行为之一）；
- 无法计算 p95/p99 尾延迟的真实分布——因为所有请求在同一时刻开始、同一时刻结束。

**v0.2 处理**：新增 `runner/loadgen.py`，支持 burst / 恒定速率 / 泊松到达，
线程池并发提交，逐请求记录 `arrival / start / ttft / finish` 四个时刻。

---

### A-03 吞吐分母口径

**证据**：`src/benchmarks/throughput.py:88-90`

```python
total_time = global_end - global_start
throughput = total_output_tokens / total_time
```

burst 场景下这个分母正确。但一旦引入到达过程（v0.2 的目标），
`global_start` 是第一次提交的时刻，而请求是随时间陆续进来的，
分母应该是 **makespan（首个到达 → 最后一个完成）**，否则会把吞吐算低/算高，
取决于是否把空闲期算进去。

**v0.2 处理**：`aggregate_records()` 统一用 `max(finish) − min(arrival)`（见 [D-04](decisions.md#d-04-吞吐用-makespan-而非墙钟结束时间)）。

---

### A-04 显存报的是整卡总量，不是生成增量

**证据**：`src/benchmarks/memory_scan.py:41-48` + `src/metrics/gpu_monitor.py:50-53`

```python
return {
    "peak_memory_mb": self.get_peak_memory_mb(),   # 整卡已用显存
    ...
}
```

7B 模型 FP16 权重约 14 GB。无论后端如何，这 14 GB 都会出现在「峰值显存」里，
两个后端的差值被稀释，且与序列长度无关的常数项占主导。

**后果**：`03_memory_vs_sequence_length.png` 的斜率含义模糊，
「节省 79.8%」这个数字实际来自对**增量**的目测估计，而非口径明确的测量。

**v0.2 处理**：记录 `baseline_memory_mb`（引擎加载后、生成前的采样），
报告 `kv_cache_delta_mb = peak − baseline`（见 [D-05](decisions.md#d-05-显存以生成增量为准并改用-nvml)）。

---

### A-05 vLLM 的 per-request 吞吐语义错误

**证据**：`src/backends/vllm_backend.py:89-99`

```python
total_time = end_time - start_time
for output in outputs:
    ...
    results.append({
        "total_time": total_time,  # batch 模式下所有请求共享总时间
        "throughput": num_output_tokens / total_time,
    })
```

`num_output_tokens` 是**单条**请求的输出，`total_time` 是**整批**的耗时。
两者的商不是任何有物理意义的量：既不是单请求吞吐（分母错），
也不是聚合吞吐（分子错）。

**v0.2 处理**：区分 `throughput_tok_s`（单请求，`tokens / e2e`）与聚合吞吐
（`Σtokens / makespan`），并在聚合层计算，不再由后端硬塞。

---

### A-06 HF 批处理把 padding 计入输出 token

**证据**：`src/backends/naive_backend.py:131-138`

```python
generated_tokens = output[input_seq_len:]
num_output_tokens = generated_tokens.shape[0]
```

`generate` 会一直跑到 `max_new_tokens`，早停的序列后面是 padding。
`shape[0]` 恒等于 `max_new_tokens`，与真实生成长度无关。

**后果**：输出 token 数被高估 → 吞吐虚高；TPOT 计算也基于错误的 token 数。

**v0.2 处理**：用 `(generated != pad_token_id).sum()`，并再与 `max_new_tokens` 取 min 兜底。

---

## 3. 调度模拟器缺陷

### A-07 预分配整条序列，违背 PagedAttention 的前提

**证据**：`src/scheduler_sim/base_scheduler.py:49`

```python
if self.memory.allocate(req.id, req.prompt_len + req.output_len):
```

一次就把 `prompt + output` 的 KV 全部分配掉。这恰恰是**分页之前**的做法。
PagedAttention 的核心是「按需增长」：先分配 prompt 的块，解码过程中每跨过一个块边界再申请。

**后果**：模拟出的显存压力、可并发数、抢占行为全部偏离真实引擎；
尤其会让「长输出请求」的显存占用被严重高估。

**v0.2 处理**：`BlockPool.ensure()` 按 token 数增量扩容，解码每步检查一次（见 [D-06](decisions.md#d-06-调度模拟器改为块级按需分配并禁止入场抢占)）。

---

### A-08 分配失败时的无限循环

**证据**：`src/scheduler_sim/base_scheduler.py:52-54`

```python
else:
    # 内存不足，放回 waiting（vLLM 会 swap，这里简化）
    waiting.insert(0, req)
```

外层是 `while len(finished) < len(requests)`。若某个请求永远分配不到内存，
它会被反复插回队首、被反复选中、反复失败——**没有退出条件，也没有超时**。
在 `kv_pool` 偏小的配置下，脚本直接挂死。

**v0.2 处理**：三重保护——准入失败即停止准入（不循环）、解码扩容失败才抢占、
抢占超过 8 次或需求超过池容量则丢弃并计入 `dropped`。

---

### A-09 时间片被重复计数

**证据**：`src/scheduler_sim/base_scheduler.py:90-124`

```python
if req.ttft is None:
    prefill_left = prefill_needed - elapsed_prefill
    if prefill_left > remaining_time:
        continue
    else:
        req.ttft = self.current_time + prefill_left
        remaining_time -= prefill_left       # 扣减了剩余时间
...
if req.ttft is not None:
    decode_elapsed = (self.current_time + dt) - req.ttft   # 又用了完整 dt
```

`remaining_time` 被算出来却从未真正用于 decode 判断，decode 仍然按整个 `dt` 推进。
同一个时间片里，prefill 结束后的那段被用了两次。

**后果**：所有请求的完成时间偏早，吞吐虚高，且误差随「prefill 在片内结束」的频率增加。

**v0.2 处理**：重写为迭代级循环，每步只推进一次，prefill 与 decode 互斥推进。

---

### A-10 p99 索引越界 / 百分位错误

**证据**：`src/scheduler_sim/base_scheduler.py:141`

```python
"p99_ttft_ms": sorted(ttfts)[int(len(ttfts)*0.99)] if len(ttfts) > 1 else 0,
```

- `len=2` → `int(1.98)=1`，取到第 2 个（等于 max，尚可）；
- `len=100` → 索引 99，恰好最后一个（等于 max，**不是 p99**）；
- 若 `len(ttfts)` 恰好使 `int(n*0.99) == n`（浮点边界），直接 IndexError。

**v0.2 处理**：`metrics/stats.py::percentile` 纯 Python 线性插值实现，边界有单测覆盖。

---

### A-11 GPU 利用率公式单位错误

**证据**：`src/scheduler_sim/base_scheduler.py:144-145`

```python
"gpu_util_avg": sum(r.actual_output_len for r in self.completed) /
               (total_time / 1000 * self.gpu.config.decode_tok_per_ms * 1000),
```

`decode_tok_per_ms` 的单位是 token/ms，乘以 `total_time/1000`（秒）再乘 1000，
量纲是 token·s/ms，与分子（token）相除得到的时间单位混乱。结果既不可解释，
也可能 > 1（利用率超过 100%）。

**v0.2 处理**：改为 `throughput / 理论峰值吞吐`，并在 `[0, 1]` 截断。

---

### A-12 SJF 使用了 oracle 信息但未声明

**证据**：`src/scheduler_sim/sjf_scheduler.py:13`

```python
return sorted(waiting, key=lambda r: r.total_len)[:batch_size]
```

`total_len = prompt_len + output_len`，而 `output_len` 是**运行结束后才知道**的量。
真实 SJF 只能用已知信息（prompt 长度或用户声明上限）排序。

**后果**：SJF 的优势被系统性夸大，读者会误以为线上可以直接获得这个收益。

**v0.2 处理**：策略重命名为 `SJF (oracle)`，在 UI 与文档中明确标注这是上界；
同时新增 `sjf_aging`（老化）与 `priority` 作为可落地的对照。

---

### A-13 MemoryPool 重复分配会泄漏

**证据**：`src/scheduler_sim/memory_pool.py:19-20`

```python
self.allocated[request_id] = blocks_needed   # 直接覆盖，不回收旧的
self.free_blocks -= blocks_needed
```

对同一 `request_id` 再调一次 `allocate`，旧块永远不会归还。

**v0.2 处理**：`ensure()` 计算 **delta**，`release()` 用 `pop`，并有单测覆盖
「增长 / 重复申请 / 释放后归还」三种路径。

---

### A-14 死代码

**证据**：`src/scheduler_sim/base_scheduler.py:71-89` 整段注释掉的旧 `_execute_step`。

**v0.2 处理**：删除。历史保留在 git。

---

## 4. 工程与可维护性

### A-15 路径硬编码，且三处拼写不一致

**证据**（同一模型在三处不同写法）：

| 文件 | 路径 |
|---|---|
| `scripts/run_baseline.sh:13` | `/root/autodl-tmp/modelscope/qwen/Qwen2___5-7B-Instruct` |
| `src/benchmarks/single_request.py:63` | `/root/autodl-tmp/modelscope/hub/Qwen/Qwen2.5-7B-Instruct` |
| `scripts/generate_prompts.py:35` | `/root/autodl-tmp/modelscope/qwen/Qwen2___5-7B-Instruct` |
| `src/benchmarks/memory_scan.py:60` | `/root/autodl-tmp/modelscope/qwen/Qwen2___5-7B-Instruct` |
| `src/benchmarks/prefix_caching.py:136` | `/root/autodl-tmp/modelscope/qwen/Qwen2___5-7B-Instruct` |

**后果**：换一台机器必须改 5 个文件；漏改一处就出现「找不到模型」。
另外 README 的快速开始引用了 `env/setup.sh`，该文件**根本不存在**。

**v0.2 处理**：统一走 `core/paths.py` 与 `ModelConfig.resolve_path()`，
支持 `LLMBENCH_MODEL_PATH` 环境变量与 `--model-path`，缺省时给出可操作的报错。

---

### A-16 依赖声明缺失

`README.md` 提到 `requirements.txt`、`env/setup.sh`，两者都不存在。
项目也没有 `pyproject.toml` / `setup.py`，无法 `pip install`，无法声明版本约束。
`vLLM 0.5.5` + `transformers 4.44.2` 只出现在文档表格里，没有任何机器可读的约束。

**v0.2 处理**：`pyproject.toml`（含 extras：`gpu`/`viz`/`dev`）、
`requirements.txt`、`requirements-gpu.txt`、`requirements-viz.txt`、`requirements-dev.txt`。

---

### A-17 无测试、无 CI、无 LICENSE

README 徽章写着 MIT，仓库里没有 `LICENSE` 文件（法律上等于未授权）。
零测试、零 CI，任何重构都无法验证是否破坏行为。

**v0.2 处理**：MIT `LICENSE`、120 个 pytest 用例、GitHub Actions（lint + 3 个 Python 版本 + 端到端流水线）。

---

### A-18 README 与 .gitignore 自相矛盾

README 第 41 行：「所有原始数据均已开源（`data/results/`）」；
`.gitignore` 第 17 行：`data/results/`。

实际仓库里**没有任何原始结果文件**，只有 7 张 PNG。读者无法复核任何一个数字。

**v0.2 处理**：结果目录按 run 存储并保留可选的 `data/results/published/`；
仓库内提交 `reports/demo/`（含生成脚本与完整 Markdown 报告）作为可复核样例。

---

### A-19 `--output` 传纯文件名会崩溃

**证据**：`src/benchmarks/single_request.py:73`

```python
os.makedirs(os.path.dirname(args.output), exist_ok=True)
```

`--output result.json` → `os.path.dirname` 返回 `""` → `makedirs("")` 抛 `FileNotFoundError`。

**v0.2 处理**：所有落盘走 `Path.parent.mkdir(parents=True, exist_ok=True)`，空父目录安全。

---

### A-20 shell 脚本缺 `pipefail`，失败被吞掉

**证据**：`scripts/run_baseline.sh:2` 与 `:87`

```bash
set -e
...
} 2>&1 | tee -a "$LOG_FILE"
```

`set -e` 对管道**不生效**（缺 `set -o pipefail`），管道的退出码是 `tee` 的（恒为 0）。
Python 脚本崩溃时，shell 继续往下跑，最终输出「Baseline tests completed!」。

**v0.2 处理**：全部改为 `set -euo pipefail`，并直接调用 `python -m llmbench`（不再手工拼参数）。

---

### A-21 每个脚本重复一遍显存清理逻辑

`throughput.py:26-43`、`latency_breakdown.py:17-22`、`memory_scan.py:17-22`、
`prefix_caching.py:14-21`、`single_request.py`（无）——四份 `gc.collect + empty_cache + sleep`
实现各不相同（sleep 3s / 5s / 10s / 15s），且都以「猜」的方式等待驱动回收。

**v0.2 处理**：收敛到 `runner/executive.py::release_gpu()` 与 `Backend.close()`，
由后端自己负责释放（vLLM 显式 shutdown engine）。

---

## 5. 可视化与报告

### A-22 图表依赖代理指标

`report_generator.py:68-85` 的 `plot_ttft_vs_prompt_length` 过滤 `output_len == 1`，
即「只画 1-token 请求」，因为只有这条数据有 TTFT 字段。
一旦补上真实 TTFT，这个过滤条件就成了错误的约束。

**v0.2 处理**：TTFT 是每条请求的一等字段，图表不再需要特殊过滤。

### A-23 中文字体在无字体环境渲染成方块

`report_generator.py:9` 写死 `SimHei` 作为首选字体。CI/服务器上通常没有该字体，
matplotlib 会退化为 DejaVu Sans 并把中文画成 `□□□`，且**不报错**。

**v0.2 处理**：`viz/theme.py` 检测系统是否有 CJK 字体；
有则中文，无则自动切换英文标签（并保留 `--lang` 手动指定）。Plotly 侧由浏览器渲染，不受影响。

### A-24 无交互、无对比视图

只有 7 张静态 PNG，无法按后端/并发筛选，无法查看原始数据。

**v0.2 处理**：自包含 HTML（Plotly，内联 JS）+ Streamlit 仪表盘（7 个页签）+ Markdown 报告；
三者共用 `Analysis` 层，口径一致。

### A-25 图表与数据可能脱节

报告中的关键倍数（1.21× / 1.61× / 79.8% / 1.75×）是**手写**在 Markdown 里的，
没有任何代码从 `data/results/` 重新计算并校验。

**v0.2 处理**：所有 KPI 由 `Analysis.kpis()` 计算并注入报告模板；
报告生成即校验（缺失数据会显式显示「无数据」而不是崩或静默）。

---

## 6. 问题清单汇总

| ID | 级别 | 位置（v0.1.0） | 影响 | v0.2 处理 |
|---|---|---|---|---|
| A-01 | P0 | `benchmarks/latency_breakdown.py:99-112` | TTFT 失真 | 后端内部计时 |
| A-02 | P0 | `benchmarks/throughput.py:18-24` | 并发曲线无效 | 负载发生器 |
| A-03 | P1 | `benchmarks/throughput.py:88-90` | 非 burst 下吞吐偏差 | makespan 口径 |
| A-04 | P0 | `metrics/gpu_monitor.py:50-53` | 显存对比不公平 | baseline + 增量 |
| A-05 | P1 | `backends/vllm_backend.py:89-99` | 单请求吞吐无意义 | 分离两种吞吐 |
| A-06 | P1 | `backends/naive_backend.py:131-138` | 输出 token 高估 | 按 pad 过滤 |
| A-07 | P0 | `scheduler_sim/base_scheduler.py:49` | 模拟偏离真实 | 块级按需分配 |
| A-08 | P0 | `scheduler_sim/base_scheduler.py:52-54` | 可能挂死 | 停止准入 + 丢弃 |
| A-09 | P0 | `scheduler_sim/base_scheduler.py:90-124` | 完成时间偏早 | 重写迭代循环 |
| A-10 | P1 | `scheduler_sim/base_scheduler.py:141` | p99 错误/越界 | 统一 percentile |
| A-11 | P2 | `scheduler_sim/base_scheduler.py:144-145` | 利用率不可解释 | 重新定义 |
| A-12 | P2 | `scheduler_sim/sjf_scheduler.py:13` | 收益被夸大 | 标注 oracle + 新策略 |
| A-13 | P2 | `scheduler_sim/memory_pool.py:19-20` | 重复分配泄漏 | delta 分配 |
| A-14 | P2 | `scheduler_sim/base_scheduler.py:71-89` | 死代码 | 删除 |
| A-15 | P2 | 5 处硬编码路径 | 换机即失败 | 统一配置 |
| A-16 | P2 | 仓库根目录 | 无法安装 | pyproject + requirements |
| A-17 | P2 | 仓库根目录 | 无授权/无回归保护 | LICENSE + 测试 + CI |
| A-18 | P2 | `README.md:41` vs `.gitignore:17` | 承诺与事实不符 | 按 run 存储 + 样例 |
| A-19 | P2 | `benchmarks/single_request.py:73` | `--output` 崩溃 | 安全 mkdir |
| A-20 | P2 | `scripts/*.sh` | 失败被吞 | `set -euo pipefail` |
| A-21 | P2 | 4 个 benchmark 文件 | 清理逻辑重复 | 收敛到 executive |
| A-22 | P2 | `utils/report_generator.py:68-85` | 图表依赖代理指标 | 一等字段 |
| A-23 | P2 | `utils/report_generator.py:9` | 中文变方块 | 字体探测 + 语言回退 |
| A-24 | P3 | 全项目 | 无交互 | HTML + Dashboard |
| A-25 | P2 | `reports/benchmark_report.md` | 数字手写不可核 | KPI 由代码计算 |

---

## 7. 值得肯定的部分

审计不是为了否定。v0.1 有几点判断是对的，v0.2 全部保留：

1. **实验矩阵设计合理**：单请求 → 并发 → 显存 → 前缀缓存 → 调度，覆盖了推理优化的完整链路。
2. **注意到「前缀缓存要测多轮对话」**：共享前缀是 prefix caching 唯一有意义的场景。
3. **手写调度模拟器**：这是最有价值的部分——它把「调度策略」从黑盒变成了可实验对象。
   v0.2 保留了这一设计并修补了实现。
4. **对显存回收有意识**：虽然实现粗糙，但「vLLM 会占住显存不还」这个坑踩对了。

---

## 8. 未在 v0.2 处理的问题

诚实记录，避免读者高估当前状态：

| 项 | 原因 | 去向 |
|---|---|---|
| 真实 GPU 上的端到端验证 | 当前环境无 24GB 显卡 | [roadmap](roadmap.md#3-gpu-验收清单) 的验收清单 |
| vLLM 各版本 API 漂移 | 无多版本环境 | 代码内做了三段回退 + 明确报错 |
| chunked prefill / 投机解码建模 | 模拟器复杂度收益比不划算 | roadmap P2 |
| 多进程压测（绕开 GIL） | 单机线程已够用 | roadmap P2 |
| 功耗/能效指标的物理校验 | 缺功率计 | 已采集，未标定 |

---

**相关文档**：[决策记录](decisions.md) · [路线图](roadmap.md) · [测量方法学](methodology.md) · [架构](architecture.md)
