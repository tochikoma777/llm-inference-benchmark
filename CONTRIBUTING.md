# 贡献指南

感谢你愿意改进这个项目。下面是让改动顺利合并的几个约定。

## 环境

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[viz,dev]"
pre-commit install          # 可选，但强烈建议
```

## 提交前自检

```bash
make fmt        # ruff --fix + black
make lint       # ruff + mypy
make test       # pytest
make demo       # 确认端到端产物仍可生成
```

CI 会重复执行这些命令。本地全绿再开 PR，能省一轮来回。

## 代码约定

- **类型注解**：公开函数必须有完整签名。项目要求 Python ≥ 3.10，可直接用 `X | None`。
- **日志**：库代码用 `llmbench.core.logging.get_logger(...)`，不要 `print`。
  `print` 只出现在 CLI 的最终输出里。
- **不吞异常**：除了清理路径与可选依赖探测，不要写裸 `except: pass`。
- **路径**：不要硬编码路径，统一走 `llmbench.core.paths`（支持环境变量覆盖）。
- **依赖**：核心包只依赖 numpy/pandas/pyyaml/matplotlib/rich。
  torch、vLLM、streamlit 必须懒加载，保证 `pip install -e .` 之后 CPU 环境也能用。

## 新增一个基准场景

1. 在 `src/llmbench/runner/scenarios.py` 里写 `run_xxx(config, backend, emit)`。
2. 每条记录至少包含 `scenario` / `level` / `bucket` / `concurrency` 与延迟字段。
   聚合逻辑由 `aggregate_records()` 统一负责，场景本身**不要**自己算 p95。
3. 在 `runner/executive.py` 的 `SCENARIOS` 与 `_dispatch` 注册。
4. 在 `core/config.py` 的 `ScenarioConfig` 增加开关。
5. 补一张图（`viz/plotly_charts.py` + `viz/figures.py`）和对应单测。

## 新增一个后端

见 README 的 [如何接入新的推理后端](../README.md#如何接入新的推理后端)。
要点：`generate()` 必须返回真实的 per-request TTFT；引擎必须在 `close()` 里释放显存。

## 测试约定

- 纯逻辑（统计、块分配、策略、聚合）必须有单测，且不依赖 GPU。
- 需要 GPU 的测试加 `@pytest.mark.gpu`，默认不会被 CI 执行。
- 端到端测试用 mock 后端，`time_scale=0` 避免真实等待。

## 文档

改了 CLI、配置项或产物路径，请同步更新 README 对应段落——文档过期比没有文档更糟。

文档分工与「什么改动该更新哪份文档」的约定见 [docs/README.md](docs/README.md)。
其中两条硬性要求：

- 新的架构决策 → `docs/decisions.md` 追加编号条目，**不要改写历史条目**；
- 新发现的旧版本缺陷 → `docs/audit-v0.1.md` 追加，并标注「审计后补」。

## 提交信息

沿用 [Conventional Commits](https://www.conventionalcommits.org/)：

```
feat(runner): measure TTFT via streaming callback
fix(scheduler): avoid livelock when the KV pool is tight
docs: document the mock backend calibration
```
