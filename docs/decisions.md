# 决策记录（ADR）

> 记录 v0.2 重构过程中每一个**影响架构或结论**的决策：为什么这么选、放弃了什么、
> 怎么验证、落在哪次提交。后续维护者可以据此判断「能不能改」以及「改了要同步什么」。

**格式**：每条编号 `D-xx`，包含 *背景 / 选项 / 决策 / 理由 / 代价 / 验证 / 关联*。

**状态图例**：✅ 已落地并验证 · 🟡 已落地待 GPU 验证 · ⏭ 已决策未实施

---

## 索引

| ID | 决策 | 状态 |
|---|---|---|
| [D-01](#d-01-整体重写为可安装包而非增量修补) | 整体重写为可安装包 | ✅ |
| [D-02](#d-02-把计时职责下沉到后端) | 计时职责下沉到后端 | 🟡 |
| [D-03](#d-03-引入负载发生器与三种到达过程) | 负载发生器 + 到达过程 | ✅ |
| [D-04](#d-04-吞吐用-makespan-而非墙钟结束时间) | 吞吐用 makespan | ✅ |
| [D-05](#d-05-显存以生成增量为准并改用-nvml) | 显存用增量 + NVML | ✅ |
| [D-06](#d-06-调度模拟器改为块级按需分配并禁止入场抢占) | 块级按需分配 | ✅ |
| [D-07](#d-07-引入模拟后端mock并明确标定与防误用) | 模拟后端 | ✅ |
| [D-08](#d-08-结果存储按运行分目录jsonl--metadata) | 按运行分目录存储 | ✅ |
| [D-09](#d-09-只做一次聚合再由多个前端消费) | 单一聚合层 | ✅ |
| [D-10](#d-10-四种交付形态共用一套图表构建函数) | 共用图表函数 | ✅ |
| [D-11](#d-11-配置用-dataclass--覆盖链--前置校验) | 配置体系 | ✅ |
| [D-12](#d-12-支持分组目录下的运行发现) | 分组运行发现 | ✅ |
| [D-13](#d-13-跨运行不合并统计重复运行取均值) | 跨运行统计规则 | ✅ |
| [D-14](#d-14-python-310--现代类型语法) | Python 3.10+ | ✅ |
| [D-15](#d-15-暂不拆分为独立仓库) | 暂不拆分为独立仓库 | ✅ |

---

## D-01 整体重写为可安装包而非增量修补

**背景**
v0.1 是 6 个独立脚本 + 4 个共用模块，没有包结构、没有测试、路径硬编码。
审计出的 25 个问题分布在**每一个文件**里（见 [audit](audit-v0.1.md)）。

**选项**

1. 就地打补丁：改 TTFT 实现、修调度器 bug，其余保留。
2. 整体重写为 `src/llmbench/` 包，保持对外行为（同样是「跑 benchmark、出图」）。

**决策**：选 2。

**理由**

- P0 问题（A-01/A-02/A-04）都要求**改动调用链的形状**：TTFT 必须在后端内部产生，
  并发必须由负载发生器驱动。补丁会在脚本层堆出一层又一层适配代码。
- 五个场景共享同一套「加载引擎 → 测量 → 落盘」流程，原本复制了 5 遍；
  重写后才能收敛成 `executive.run_suite()`。
- 可安装包才能让 CI、`pip install -e .`、第三方复用成为可能。

**代价**

- 破坏性变更：旧入口全部失效，需要迁移说明（CHANGELOG 已写）。
- 一次性投入大，短期内没有「可运行」的中间状态。

**验证**：`pip install -e .` → `llmbench run -c config/smoke.yaml` 全流程通过；
120 个测试覆盖新结构。

**关联**：提交 `88eeecf`；`src/llmbench/**`、`pyproject.toml`。

---

## D-02 把计时职责下沉到后端

**背景**
A-01 显示 TTFT 只能靠「另起一次请求」代理。根本原因是**运行器不知道第一个 token
什么时候出现**——它只能在请求返回后拿到总耗时。

**选项**

1. 运行器轮询（在外部线程里查输出长度）——需要引擎支持增量查询，通用性差。
2. 后端返回逐 token 时间戳，运行器只做算术。

**决策**：选 2。`GenerationOutput` 携带 `ttft_ms`、`token_times_ms`、`e2e_ms`。

**理由**

- 只有引擎内部能准确知道「第一个 token 可用」的时刻。
- HF 侧的 `StoppingCriteria` 天然在每步解码后回调，零额外成本；
  vLLM 侧本来就是流式接口，打点即用。
- 计时与引擎实现解耦：新后端只要如实上报，其余能力（图表、报告）自动获得。

**代价**

- 每个后端都要实现计时，新接入方必须理解契约（已写进 CONTRIBUTING 与 base.py docstring）。
- vLLM 的离线批量模式拿不到 per-request TTFT，只能回退到 `RequestOutput.metrics`
  或标注 `ttft_source=e2e_proxy`。

**验证**：`tests/test_backends.py` 校验 `ttft ≤ e2e`、`tpot = (e2e−ttft)/(n−1)`；
mock 后端在 `time_scale=0` 下时间关系仍然自洽。

**关联**：提交 `88eeecf`；`backends/base.py`、`backends/naive_backend.py`、
`backends/vllm_backend.py`。

---

## D-03 引入负载发生器与三种到达过程

**背景**
A-02：原「并发测试」是一次 `generate_batch`，没有时间维度。

**选项**

1. 保持批处理，只把 `concurrency` 改名为 `batch_size`——诚实但放弃了核心能力。
2. 引入线程池 + 到达调度，逐请求计时。

**决策**：选 2，并支持三种到达模式。

| 模式 | 语义 | 用途 |
|---|---|---|
| `burst` | 全部 t=0 | 吞吐上限、饱和点 |
| `constant` | 固定 1/rate 间隔 | 稳态服务能力 |
| `poisson` | 指数间隔 | 最接近线上 |

**理由**

- 排队时间（`queue_ms`）只能通过到达-准入的时间差观测，这是连续批处理的关键证据。
- 三种模式覆盖「压满 / 稳态 / 真实」三类问题，成本只是一段 60 行的调度代码。
- 线程模型与真实服务的准入队列同构（`max_workers` = 并发上限）。

**代价**

- Python GIL 限制：数百并发时线程切换会成为噪声。已在 `methodology.md` 声明，
  多进程方案列入 roadmap。
- `time_scale=0` 时模拟后端不会真正并发，测试需显式给非零缩放（已在测试注释中说明）。

**验证**：`tests/test_backends.py::test_run_load_records_all_fields`、
`test_plan_arrivals_*`；demo 中 c=1→8 的吞吐曲线呈亚线性增长（符合物理预期）。

**关联**：提交 `88eeecf`；`runner/loadgen.py`。

---

## D-04 吞吐用 makespan 而非墙钟结束时间

**背景**
A-03：一旦有到达过程，`wall_time` 取决于第一个请求何时提交，把服务空闲期也算进分母。

**选项**

1. `Σtokens / wall_time`（简单，但受测试脚手架影响）。
2. `Σtokens / (max(finish) − min(arrival))`（makespan）。

**决策**：选 2。burst 下退化为 `max(e2e)`，与旧口径一致，保证可比。

**理由**

- makespan 是服务视角的「这段负载用了多久」，与客户端何时按下回车无关。
- 同一份数据在 burst 与 poisson 下都成立，避免出现「换个到达模式吞吐就变了」的假象。

**代价**：需要每条记录都带 `arrival_ms` / `finish_ms`（schema 稍宽）。

**验证**：`tests/test_results.py::test_aggregate_computes_makespan_throughput`
用两条 1s/2s 的请求断言 64 tok/s。

**关联**：提交 `88eeecf`；`core/results.py::aggregate_records`。

---

## D-05 显存以生成增量为准，并改用 NVML

**背景**
A-04：整卡显存包含 14GB 权重，掩盖真实差异。原实现每次采样 fork 一次 `nvidia-smi`（~40ms）。

**选项**

1. 继续读整卡显存，人工减去权重（需要知道权重大小，且量化后不确定）。
2. 记录「引擎加载完成、尚未生成」的基线，报告峰值与基线之差。
3. 用 `torch.cuda.memory_allocated()`（看不到 vLLM 自定义分配器与 CUDA graph）。

**决策**：选 2；采样源从 `nvidia-smi` 换成 NVML（`pynvml`），后者仅作回退。

**理由**

- 基线差值就是「这次生成额外占用的显存」，跨后端可直接比较，且不依赖权重精度。
- NVML 单次读取 < 1ms，可支持 100ms 采样而不干扰基准；还顺带拿到功耗与温度。
- `memory_allocated()` 只反映 PyTorch 分配器视角，对 vLLM 无意义。

**代价**：基线必须在引擎 `load()` 之后、第一次 `generate()` 之前采样，顺序错了结果就错。
`Measurement` 上下文管理器强制了这个顺序。

**验证**：`tests/test_reporting.py` 校验 `memory_saving_pct` 落在 (0,100)；
demo 中 2048 token 时增量 416MB vs 2058MB，比例 79.8%，与原报告一致。

**关联**：提交 `88eeecf`；`metrics/gpu_monitor.py`、`runner/measure.py`。

---

## D-06 调度模拟器改为块级按需分配，并禁止入场抢占

**背景**
A-07/A-08/A-09 三条 P0：预分配整条序列、失败死循环、时间片重复计数。

**选项**

1. 修 bug，保留「一次分配 prompt+output」。
2. 重写为迭代级循环：按需扩容、可抢占、有收敛保护。

**决策**：选 2，并新增两条硬规则：

- **等待队列永不抢占运行队列**（对齐 vLLM 默认策略）。
- **不可行请求直接丢弃**：需求 > 池容量，或抢占超过 8 次。

**理由**

- 预分配会让长输出请求的显存占用被高估数倍，直接扭曲「可并发数」与「抢占频率」。
- 入场抢占是活锁的根源：A 抢 B、B 抢 A，模拟 10 万步不收敛（修复前实测 99,400 次抢占）。
- 真实系统遇到装不下的请求会一直排队或报错，而不是把别的请求踢掉。

**代价**

- 模型更复杂（约 220 行），需要单测覆盖抢占/丢弃/收敛三条路径。
- 与真实 vLLM 仍有差距：未建模 chunked prefill、swap、recompute 的 CPU 开销。

**验证**：`tests/test_scheduler.py` 11 个用例，含「5 种策略全部完成」「SJF 改善短请求
但恶化长尾」「极小池不挂死」「不可扩容请求被丢弃」。

**关联**：提交 `88eeecf`；`scheduler_sim/{engine,model,policies}.py`。

---

## D-07 引入模拟后端（mock），并明确标定与防误用

**背景**
重构后所有能力（报告、仪表盘、CI）都依赖「有结果」才能验证；而 GPU 不是随处可得。

**选项**

1. 用真实数据做 fixture（把 3090 的结果提交进仓库）——只能测渲染，测不到采集链路。
2. 写一个**解析式**后端：用标定过的公式模拟延迟与显存，走完整代码路径。
3. 随机数后端——简单但产生无意义的曲线。

**决策**：选 2。三种服务模式各有独立的标定参数。

**理由**

- 走完整路径意味着 CI 能验证「负载 → 遥测 → 聚合 → 图表 → HTML」的每一环。
- 解析模型可以让「连续批处理为何更快」这件事在代码里可读，而不是黑盒。
- 参数来自 v0.1 的真实测量（39/47 tok/s、321 tok/s @ c=8、79.8% KV 节省），
  在并发 8 下复现出 308/229/86 tok/s，与实测同量级。

**代价 / 风险**

- **最大风险是被误当成实测数据**。缓解措施：
  - `metadata.system.mock = true` 贯穿到存储层；
  - `Analysis.is_mock` 驱动报告顶部横幅与仪表盘警示条；
  - 报告标题与 Markdown 首段显式声明；
  - 文档中反复标注「不可引用绝对性能」。

**验证**：`tests/test_backends.py` 校验三种模式的吞吐排序、前缀缓存加速、
遥测清零；`test_reporting.py` 校验 mock 报告带警示。

**关联**：提交 `88eeecf`；`backends/mock_backend.py`、`docs/methodology.md` §7。

---

## D-08 结果存储按运行分目录：JSONL + metadata

**背景**
A-18：原结果目录被 gitignore，扁平 JSON 无环境信息，数字无法复现。

**选项**

1. 单个 `results.csv`（简单，但无法承载嵌套字段与逐请求记录）。
2. 每次运行一个目录：`metadata.json` + `results.jsonl` + `summary.json`。

**决策**：选 2。

**理由**

- 逐请求记录数量大且异构（不同场景字段不同），JSONL 天然适配且可逐行排查。
- `metadata.json` 存环境指纹与配置快照，是「可审计」的前提——没有它，一个数字毫无意义。
- 目录即运行，删除/对比/归档都是文件操作，不需要数据库。

**代价**：每次运行产生 3 个文件；聚合需要在读入时做一次（毫秒级）。

**验证**：`tests/test_results.py` 覆盖写入、失败标记、损坏运行跳过、分组发现。

**关联**：提交 `88eeecf`；`core/results.py`。

---

## D-09 只做一次聚合，再由多个前端消费

**背景**
v0.1 的 p95 分散在「模拟器内部公式」「绘图脚本的 pandas 逻辑」「手写报告」三处，
口径不可能一致（A-10/A-25）。

**选项**

1. 每个前端各自算——灵活但必然漂移。
2. 场景只产出原子记录，`aggregate_records()` 统一计算，前端只读结果。

**决策**：选 2。

**理由**

- 一条测量只有一种 p95，图表与表格不可能打架。
- 新增图表不需要重算指标；新增指标只需改一处。
- 聚合规则（分组键、makespan、失败过滤）可以集中测试。

**代价**：场景作者必须理解 `bucket` 的语义（已在 CONTRIBUTING 写明）。

**验证**：`tests/test_results.py` 4 个聚合用例；
`test_reporting.py::test_repeated_runs_are_averaged_not_summed`。

**关联**：提交 `88eeecf`；`core/results.py`、`reporting/analysis.py`。

---

## D-10 四种交付形态共用一套图表构建函数

**背景**
需要同时产出：交互式 HTML、Streamlit 仪表盘、静态 PNG（论文/幻灯片）、Markdown（GitHub）。

**选项**

1. 每种形态各写一套图表代码（4 倍工作量，且必然不一致）。
2. 以 Plotly 图对象为中间层，HTML 与 Streamlit 直接复用；matplotlib 单独实现但共享主题。

**决策**：选 2，接受 matplotlib 侧的一份平行实现。

**理由**

- HTML 与 Streamlit 的渲染模型完全一致，复用成本为零。
- matplotlib 的排版能力（字体、DPI、矢量导出）不可替代，强行统一会两边都别扭。
- 共享的是**主题层**（配色、字号、语义色）与**数据层**（`Analysis`），而非绘图 API。

**代价**：新增一张图要写两份（Plotly + matplotlib），容易只写一份。
缓解：`make_all_figures` 对失败图只告警不中断，README 明确「HTML 为主，PNG 为导出」。

**验证**：`test_reporting.py::test_figures_are_rendered` 断言 ≥5 张图且非空。

**关联**：提交 `88eeecf`；`viz/plotly_charts.py`、`viz/figures.py`、`viz/theme.py`。

---

## D-11 配置用 dataclass + 覆盖链 + 前置校验

**背景**
v0.1 的配置散落在 `argparse` 默认值、shell 变量、模块常量、README 表格四处，
且互相矛盾（A-15）。

**选项**

1. 纯 dict + `get()`（灵活，拼错键静默变 None）。
2. dataclass 树 + YAML/env/`--set` 覆盖 + `validate()`。

**决策**：选 2。覆盖顺序：**dataclass 默认 → YAML → 环境变量 → `--set`**。

**理由**

- 拼错的键会进入 `extra`（而不是消失），既宽松又可见。
- 校验发生在加载引擎之前：并发为 0、未知后端这类错误 0.1 秒内失败，
  而不是等 14GB 权重加载完才崩。
- 配置快照随结果落盘，重跑只需 `-c` 指向保存的文件。

**代价**：新增配置项要改三处（dataclass、默认 YAML、必要时校验）。

**验证**：`tests/test_config.py` 15 个用例，覆盖嵌套覆盖、env、非法值、未知键。

**关联**：提交 `88eeecf`；`core/config.py`、`config/*.yaml`。

---

## D-12 支持分组目录下的运行发现

**背景**
`make demo` 把三次运行写入 `data/results/demo/<run>/`，而存储层只扫一层，
导致 `llmbench runs`、`report --latest`、仪表盘全部「看不到」演示数据。

**选项**

1. 让 demo 直接写 `data/results/`——会与真实实验混在一起。
2. 存储层支持一层分组（`<root>/<group>/<run>/`）。

**决策**：选 2。

**理由**：分组是自然的组织方式（按模型、按硬件、按实验批次），
支持它比强迫用户拍平目录更合理。

**代价**：`run_dirs()` 需要两次 glob 并去重；深层嵌套仍不支持（刻意限制，避免误扫）。

**验证**：`tests/test_results.py::test_store_discovers_grouped_runs`。

**关联**：提交 `d6190a3`；`core/results.py::run_dirs`。

---

## D-13 跨运行不合并统计，重复运行取均值

**背景**
修复 D-12 后发现新问题：合并多次运行做报告时，同一配置的 3 次运行被合并进同一个
聚合桶，输出 token 数相加而 makespan 不变，吞吐被放大 3 倍。

**选项**

1. 聚合键包含 `run_id`，图表直接展示多点。
2. 聚合键包含 `run_id`，展示时按 (series, concurrency) 取均值。

**决策**：选 2。

**理由**

- makespan 只在单次运行内有意义，跨运行相加是量纲错误——必须在聚合层就隔离。
- 同一配置跑 3 次是「重复测量」，图表上应是一个点（可带误差棒），而不是三个点。

**代价**：`concurrency_frame()` 等视图需要做一次 groupby-mean；
`n_runs` 列记录参与平均的运行数，便于 UI 标注。

**验证**：`test_aggregate_never_merges_different_runs`（单测）、
`test_repeated_runs_are_averaged_not_summed`（端到端）；
连续执行两次 `make demo`，峰值吞吐稳定在 308 tok/s（修复前会变成 616 / 925）。

**关联**：提交 `88eeecf`；`core/results.py`、`reporting/analysis.py::_collapse_runs`。

---

## D-14 Python 3.10+ 与现代类型语法

**背景**
代码中大量使用 `typing.Dict/List/Optional`，ruff 的 `UP` 规则建议改为内置泛型与 `X | None`。

**选项**

1. 保持 3.9 兼容，压制 `UP` 规则。
2. 提升最低版本到 3.10，全面使用现代写法。

**决策**：选 2。

**理由**

- 3.9 已于 2025 年 10 月 EOL；vLLM 与主流生态也早已要求 ≥3.9，实践中 3.10 是下限。
- `X | None` 在 `get_type_hints()` 下需要 3.10+（配置构造依赖它做运行时反射）。
- 少一层 `typing` 导入，可读性更好。

**代价**：放弃 3.9 用户；CI 覆盖 3.10–3.12。

**验证**：`ruff check` 零告警、`mypy` 通过、CI 三版本矩阵。

**关联**：提交 `88eeecf`；`pyproject.toml`。

---

## D-15 暂不拆分为独立仓库

**背景**
重构后包名是 `llmbench`，与仓库名 `llm-inference-benchmark` 不一致，存在「是否该独立成库」的疑问。

**决策**：**暂不拆分**，在原仓库以分支 + PR 的方式发布 v0.2。

**理由**

1. **叙事完整性**：v0.1 → v0.2 的演进本身就是这个项目最有价值的部分——
   它展示了「发现问题 → 定位根因 → 重构验证」的完整工程判断。拆成两个仓库会把这个故事切断。
2. **可追溯性**：审计文档里的每一条都能用 `git show v0.1.0:...` 定位到原始代码。
   拆库后这条链路断裂。
3. **维护成本**：两处 issue、两套 CI、两份 README，对小项目是纯负担。
4. **包名不等于仓库名**：`pip install llmbench` 与仓库名解耦是常态（如 `requests` / `psf/requests`）。

**何时应该拆**

- 当它被第三方**作为库**依赖（而不只是阅读/复用思路）；
- 当需要独立发版节奏（例如库每月发版、benchmark 结果每季度更新）；
- 当 benchmark 数据量增长到需要独立仓库托管（如 `results` 分支或 LFS）。

满足任一条件时再拆，并把本仓库保留为「参考实现 + 历史数据」。

**关联**：见 [roadmap](roadmap.md#4-发布与协作流程)。

---

## 开发过程记录

按时间顺序，便于溯源「什么时候做了什么、为什么」。

| 阶段 | 动作 | 产出 | 验证 |
|---|---|---|---|
| 1 | 克隆并逐文件审计 v0.1 | 25 条问题（[audit](audit-v0.1.md)） | 与 `git show v0.1.0` 对照 |
| 2 | 工程化骨架：venv、pyproject、LICENSE、requirements | 可 `pip install -e .` | `llmbench info` |
| 3 | 核心层：config / paths / logging / results / stats / 遥测 | `core/`、`metrics/` | 单元测试 |
| 4 | 后端层：协议 + mock + HF + vLLM | `backends/` | mock 三模式对比 |
| 5 | 运行层：workload / loadgen / measure / scenarios / executive | `runner/` | 冒烟跑通五场景 |
| 6 | 调度模拟器重写 | `scheduler_sim/` | 11 个单测 |
| 7 | 可视化与报告 | `viz/`、`reporting/` | 8 张图 + HTML/MD |
| 8 | Streamlit 仪表盘 | `dashboard/app.py` | `AppTest` 7 页签 |
| 9 | CLI + 脚本 + 文档 | `cli.py`、`scripts/`、`docs/` | 端到端 |
| 10 | 质量闸门：ruff / black / mypy / pytest / CI | — | 120 passed |
| 11 | 演示产物 + 版本标签 | `reports/demo/`、`v0.1.0`/`v0.2.0` | `make demo` 幂等 |

---

**相关文档**：[审计](audit-v0.1.md) · [路线图](roadmap.md) · [方法学](methodology.md)
