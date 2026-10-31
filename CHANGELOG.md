# Changelog

本项目遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [0.2.1] — 2026-10-31

补丁版本：仅测试与文档修正，**无运行时行为变化**，数据格式不变（schema v2）。

### 修复

- 测试不再跨模块导入 `tests.conftest`。该写法只在 `python -m pytest` 下成立
  （其将 cwd 加入 `sys.path`），CI 使用的 `pytest` 命令会抛
  `ModuleNotFoundError: No module named 'tests'`。改为 `make_run` 工厂 fixture。
- HTML 报告的版本徽标改为读取 `llmbench.__version__`，避免与 `pyproject.toml` 脱节。

### 新增

- `CODEBUDDY.md`：仓库级协作说明（当前状态、命令、约定、陷阱、下一步），
  供新会话或新维护者快速接手。
- `docs/README.md` 的阅读路径中补充「接手 / 继续开发」入口。

## [0.2.0] — 2026-10-31

一次彻底的重构。核心动机：把「能跑出数字的脚本」升级为「结论可信、可复现、
可交付的基准测试框架」。

### 测量正确性

- **真实 TTFT**：HF 后端通过 `StoppingCriteria` 在每次解码步打点；vLLM 后端改用流式
  `AsyncLLMEngine`，逐请求记录首 token 时间。移除了「生成 1 token 当 TTFT」的代理指标。
- **真实并发**：新增负载发生器，支持 burst / 恒定速率 / 泊松到达，线程池并发提交，
  每个请求独立记录到达、入队、首 token、完成四个时刻。
- **吞吐口径修正**：吞吐 = 总 token / makespan（首个请求调度到最后一个请求完成），
  在非 burst 到达下依然正确。
- **显存口径修正**：以引擎加载后的基线显存为参考，只统计生成带来的 KV 增量。
- **遥测升级**：改用 NVML 直接读取（含功耗、温度、能量积分），`nvidia-smi` 仅作回退。

### 调度模拟器

- 修复原有的三个缺陷：p99 索引越界、预填充时间片计算错误、分配失败时的死循环。
- 改为块级按需分配（不再预分配 `prompt_len + output_len`），支持解码期扩容与抢占。
- 新增策略：优先级、老化 SJF、轮转；新增公平性（Jain 指数）、分组延迟、抢占次数指标。
- 新增不可行请求的丢弃保护，避免极小显存配置下无限抢占。
- 移除入场抢占（与 vLLM 默认行为一致），从根本上消除两个请求互相抢占的活锁。

### 工程化

- 重构为可安装包 `llmbench`（`pip install -e .`），提供 `llmbench` CLI。
- 统一的配置系统：dataclass + YAML/JSON + 环境变量 + `--set` 覆盖，带校验与错误提示。
- 结构化的结果存储：每次运行一个目录（`metadata.json` / `results.jsonl` / `summary.json`），
  记录环境指纹与配置快照。
- 新增模拟后端（`mock`），无需 GPU 即可跑通全链路，产物标记 `is_mock`。
- 移除所有硬编码路径；`data/`、`reports/`、`logs/` 均可通过环境变量重定向。

### 可视化与报告

- 新增自包含 HTML 报告（内联 Plotly，离线可用）、Markdown 报告、Streamlit 交互仪表盘。
- 统一视觉主题：语义化配色、中文字体自动降级、对数坐标只显示实测档位。
- 新增图表：效率前沿（吞吐 vs p99）、延迟 CDF、请求时间线（甘特）、调度公平性对比。
- 图表与分析共用同一份 `Analysis` 层，杜绝「图与表对不上」。

### 质量保障

- 新增 100+ 个 pytest 用例，覆盖统计、配置、存储、聚合、调度器、负载生成、报告、CLI 与仪表盘。
- 引入 ruff + black + mypy 配置，以及 GitHub Actions CI（多 Python 版本 + 端到端冒烟）。
- 补充 LICENSE、CONTRIBUTING、CHANGELOG。

### 文档与溯源

- `docs/audit-v0.1.md`：v0.1 全量审计，25 个问题按 P0/P1/P2 分级，每条带
  `文件:行号 @ v0.1.0` 证据、影响分析与 v0.2 处理方式。
- `docs/decisions.md`：15 条 ADR（背景 / 选项 / 决策 / 理由 / 代价 / 验证 / 关联提交），
  附开发过程时间线，可从任意决策回溯到具体提交。
- `docs/roadmap.md`：当前验证状态、GPU 验收清单（可执行 + 通过标准）、
  后续优化优先级、发布与协作流程、风险登记。
- `docs/README.md`：文档索引、阅读路径与维护约定。

### 破坏性变更

- 旧入口 `python src/benchmarks/*.py` 全部移除，改用 `llmbench run`。
- 结果目录结构变更：`data/results/<run_id>/` 取代扁平的 `naive_baseline.json` 等文件。
- 旧图表脚本 `src/utils/report_generator.py` 移除，改用 `llmbench report`。

## [0.1.0] — 2026-06

初始版本：单请求基线、并发吞吐、前缀缓存、延迟拆解、显存扫描，以及 FCFS/SJF
调度模拟，附带 7 张图表与一份手写技术报告（见
[docs/reference-run-rtx3090.md](docs/reference-run-rtx3090.md)）。
