# CODEBUDDY.md

> 仓库级协作说明。**人类入口是 [README.md](README.md)**；这份文件是给 AI 助手、
> 新维护者、以及"半年后的自己"的上下文压缩包。
> 会话被清空后，读这一份 + `docs/roadmap.md` 第 1 节即可继续工作。

---

## 1. 这是什么

一套可复现、可审计的 LLM 推理基准测试框架（包名 `llmbench`）。
核心主张：**TTFT 必须由引擎内部实测，而不是代理估算**；**每个数字都要能回溯到原始记录**。

对比三种服务模式（vLLM 连续批处理 / HF 静态批处理 / HF 单请求）在吞吐、
尾延迟、KV Cache 占用、前缀缓存、调度策略上的差异。

## 2. 当前状态（v0.2.1）

| 维度 | 状态 |
|---|---|
| 代码完整性 | ✅ 全部功能已实现 |
| 模拟后端端到端 | ✅ 已验证（`make demo`，120 测试，CI 全绿） |
| 真实 vLLM / HF 路径 | ⚠️ **未在 GPU 上跑过**（本机无 24GB 卡） |

**唯一阻塞项 = GPU 验收**。清单见 [`docs/roadmap.md` §3](docs/roadmap.md#3-gpu-验收清单)，
包含可执行命令、通过标准、预期数值区间、已知会踩的坑。做完后按 §3.6 更新 README 与 CHANGELOG。

> 在完成验收前，**任何绝对性能数字都不能对外引用**。README 里的"核心结论"标注了模拟来源。

## 3. 30 秒上手

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[viz,dev]"

make demo                    # 三后端演示 + 生成报告（无需 GPU）
llmbench dashboard           # 交互仪表盘
make test                    # 120 用例
make check                   # lint + test
```

真实 GPU：

```bash
pip install -r requirements-gpu.txt
export LLMBENCH_MODEL_PATH=/path/to/Qwen2.5-7B-Instruct
bash scripts/run_gpu.sh "$LLMBENCH_MODEL_PATH"
```

## 4. 代码地图

```
src/llmbench/
├── cli.py                 # 唯一入口：run/report/dashboard/sim/runs/show/export/datasets/info/…
├── core/
│   ├── config.py          # dataclass 配置树 + YAML/env/--set 覆盖 + validate()
│   ├── results.py         # RunWriter / ResultStore / aggregate_records（聚合唯一实现）
│   ├── paths.py           # 所有路径解析，支持 LLMBENCH_* 环境变量
│   └── logging.py         # rich 日志
├── backends/
│   ├── base.py            # Backend 契约：generate() 必须返回带真实 ttft 的 GenerationOutput
│   ├── mock_backend.py    # 解析式模拟（标定自 v0.1 实测），无 GPU
│   ├── naive_backend.py   # HF：StoppingCriteria 逐步打点
│   └── vllm_backend.py    # vLLM：AsyncLLM / AsyncLLMEngine 流式，失败回退 offline
├── runner/
│   ├── loadgen.py         # 到达过程（burst/constant/poisson）+ 线程池 + 四时刻记录
│   ├── scenarios.py       # 5 个场景，只产出原子记录，不算聚合
│   ├── measure.py         # 遥测上下文（基线显存 → 增量）
│   └── executive.py       # 引擎生命周期、热身、场景编排、落盘
├── scheduler_sim/         # 块级 KV 分配 + 5 种策略 + 迭代级引擎
├── reporting/             # Analysis（唯一指标入口）→ html / markdown
├── viz/                   # theme（配色/字体）· plotly_charts · figures(matplotlib)
└── dashboard/app.py       # Streamlit，复用 plotly_charts
```

**数据流**：`config → backend → loadgen → ResultStore(JSONL) → Analysis → 图表/报告`

## 5. 不可破坏的不变量

改代码前先读这几条，它们解释了为什么结构是这样（详见 [docs/decisions.md](docs/decisions.md)）：

1. **计时在后端**。`GenerationOutput.ttft_ms` 必须由引擎内部产生。
   不要在 runner 层"估算" TTFT——v0.1 就是因为这个导致结论失真（A-01）。
2. **聚合只在一处**。所有 p50/p95/p99、吞吐、makespan 都由
   `core/results.py::aggregate_records` 计算。场景与图表**不得**自己算统计。
3. **模拟数据必须可识别**。`metadata.system.mock` 必须为 true，`Analysis.is_mock`
   驱动报告横幅与仪表盘警示。任何情况下都不能去掉这个标记。
4. **路径走 `core/paths.py`**。禁止硬编码（v0.1 有 5 处硬编码且三处拼写不一致）。
5. **重依赖懒加载**。`import llmbench` 不能 import torch / vllm / streamlit，
   否则 CPU 环境无法用 `report` / `dashboard`。
6. **不移动已发布的 git tag**。审计文档的 `文件:行号` 引用依赖 `v0.1.0` 固定指向 `7b047a2`。
7. **跨运行不合并统计**。makespan 只在单次运行内有意义；重复运行展示取均值（D-13）。

## 6. 常见改动 → 必须同步什么

| 你改了什么 | 必须同步 |
|---|---|
| 测量口径（TTFT/TPOT/吞吐/显存） | `docs/methodology.md` + 相关单测 + README 数字来源说明 |
| 新增 CLI 子命令 / 配置项 | `README.md`「命令一览」「配置」+ argparse help |
| 新增场景 | `scenarios.py` + `executive.SCENARIOS` + `ScenarioConfig` + 图表 + 单测 |
| 新增后端 | 实现 `Backend` 三个方法 + `register_backend` + `available_backends()` 描述 |
| 架构决策 | `docs/decisions.md` **追加** `D-xx`（不改历史条目） |
| 发现新缺陷 | `docs/audit-v0.1.md` 追加 `A-xx` |
| 完成 roadmap 条目 | `docs/roadmap.md` 移入已完成 + `CHANGELOG.md` 记一行 |
| 发版 | `pyproject.toml` + `src/llmbench/__init__.py` 版本号 + CHANGELOG + tag |

## 7. 已知陷阱

- **`python -m pytest` ≠ `pytest`**：前者把 cwd 加进 `sys.path`，后者不加。
  测试里不要出现 `from tests.conftest import ...`（CI 就是这么红的）。用 fixture。
- **`time_scale=0` 时 mock 不会真正并发**：线程瞬间完成，观测不到批处理增益。
  测并发行为要给一个小的非零缩放（见 `tests/test_backends.py`）。
- **显存基线必须在 `backend.load()` 之后采**，否则增量会把权重算进去。
  `Measurement` 上下文已强制该顺序，别绕过它。
- **vLLM 会占住显存不还**：`close()` 里显式 shutdown + `empty_cache`；
  两个引擎之间要 `release_gpu()`。
- **无 CJK 字体的机器上 matplotlib 会画方块**：`viz/theme.py` 会自动降级成英文，
  不要硬编码 `SimHei`。Plotly/HTML 由浏览器渲染，不受影响。
- **`AGGREGATE` 行的 `bucket` 字段**：场景必须显式设置，否则聚合会与派生行重复计数。

## 8. 下一步

按优先级（完整列表见 [docs/roadmap.md](docs/roadmap.md) §5）：

1. **GPU 验收**（唯一 P0）——跑通后补 `docs/gpu-validation-<date>.md`，README 加"实测"列。
2. vLLM 版本兼容矩阵（v1 / legacy / offline 三条路径都要真机确认）。
3. `llmbench compare A B` 子命令（两次运行的 delta 表）。
4. OpenAI 服务端模式 / 多进程压测 / 采样精度自适应。

## 9. 文档地图

| 想知道 | 看 |
|---|---|
| 怎么用、结论是什么 | [README](README.md) |
| 原来的代码错在哪 | [docs/audit-v0.1.md](docs/audit-v0.1.md)（25 条，带 `文件:行号 @ v0.1.0`） |
| 为什么这样设计 | [docs/decisions.md](docs/decisions.md)（15 条 ADR） |
| 还差什么、怎么验收 | [docs/roadmap.md](docs/roadmap.md) |
| 数字怎么来的 | [docs/methodology.md](docs/methodology.md) |
| 代码怎么分层 | [docs/architecture.md](docs/architecture.md) |
| 历史实测数据 | [docs/reference-run-rtx3090.md](docs/reference-run-rtx3090.md) |
| 文档维护约定 | [docs/README.md](docs/README.md) |

## 10. 提交前自检

```bash
.venv/bin/ruff check src tests
.venv/bin/black --check src tests
.venv/bin/mypy src/llmbench
.venv/bin/pytest            # 注意用 console script，与 CI 一致
make demo                   # 确认产物仍能生成
```

全绿再提交。CI 会跑同样的四条 + 端到端流水线。
