# 路线图与验收清单

> 回答三个问题：现在到哪一步了？还差什么才能收尾？后面往哪走？

---

## 1. 当前状态（v0.2.0）

### 1.1 已完成并验证

| 能力 | 状态 | 证据 |
|---|---|---|
| 包化与 CLI | ✅ | `pip install -e .`，11 个子命令 |
| 配置体系（YAML/env/`--set` + 校验） | ✅ | `tests/test_config.py` 15 例 |
| 结果存储（按运行分目录 + 环境指纹） | ✅ | `tests/test_results.py` 12 例 |
| 负载发生器（burst/constant/poisson） | ✅ | `tests/test_backends.py` |
| 模拟后端（三种服务模式，标定） | ✅ | demo：308 / 229 / 86 tok/s @ c=8 |
| 调度模拟器（5 策略、块级分配、抢占） | ✅ | `tests/test_scheduler.py` 15 例 |
| 聚合与分析层 | ✅ | `tests/test_reporting.py` |
| 静态图 / HTML / Markdown 报告 | ✅ | `make demo` 产出 8 PNG + HTML + MD |
| Streamlit 仪表盘 | ✅ | `AppTest`：7 页签、0 异常 |
| 质量闸门 | ✅ | ruff / black / mypy / 120 tests |
| CI（lint + 3.10–3.12 + 端到端） | ✅ | `.github/workflows/ci.yml` |

### 1.2 已落地但**未在真实硬件验证**

以下代码路径只有在 24GB 级 GPU + 真实模型上才能验证：

| 模块 | 风险点 |
|---|---|
| `backends/naive_backend.py` | `StoppingCriteria` 在不同 transformers 版本的回调签名；`device_map="auto"` 与单卡/多卡行为；4bit/8bit 量化分支 |
| `backends/vllm_backend.py` | `AsyncLLM`（v1）与 `AsyncLLMEngine`（legacy）的导入与 `generate` 签名；流式迭代器是否逐 chunk 产出；`RequestOutput.metrics` 字段名在版本间的差异 |
| `metrics/gpu_monitor.py` | NVML 在多进程/多卡下的设备索引；功耗字段在某些消费级卡上不可读 |
| 端到端 | 显存是否真的被释放（vLLM 的 `shutdown` 行为）；长时间运行的稳定性 |

> **这正是当前唯一阻塞项**。功能实现完整、mock 全程可跑，但「真实引擎的 API 兼容性」
> 只能在有卡时确认。下面给出可直接执行的验收清单。

---

## 2. 收尾判断

**结论：可以收尾，但必须区分「功能完成」与「硬件验证完成」。**

- 作为**可运行、可展示、可协作的工程项目**：已可收尾。
  任何人在无 GPU 机器上 `make demo` 就能看到完整产物，CI 也在持续验证。
- 作为**可引用的性能结论**：尚未收尾。README 中所有数字都标注了「模拟」，
  真实数字需要跑一遍验收清单后补上。

建议的收尾动作：

1. 合并到 `main` 并打 `v0.2.0` 标签（当前已是本地提交，待 push）。
2. README 中的「核心结论」表格保留模拟数据标题，等 GPU 验证后再新增一列真实数据。
3. 把第 3 节的验收清单作为下一位维护者（或未来的自己）的入口。

---

## 3. GPU 验收清单

目标：在一张 ≥24GB 的卡上，用约 40 分钟确认所有真实路径可用，并产出可引用的数字。

### 3.1 前置

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[viz,dev]"
pip install -r requirements-gpu.txt        # torch / transformers / vllm

nvidia-smi                                  # 确认 ≥24GB 可用
llmbench info                               # 确认 torch/vllm 版本被识别
```

**通过标准**：`llmbench info` 中 `package:torch`、`package:vllm` 均非 `not installed`。

### 3.2 模型

```bash
llmbench fetch-model --model qwen/Qwen2.5-7B-Instruct --cache-dir ~/models
export LLMBENCH_MODEL_PATH=~/models/qwen/Qwen2.5-7B-Instruct
```

**通过标准**：目录内存在 `config.json` 与 `*.safetensors`。

### 3.3 引擎冒烟（最关键的一步）

```bash
llmbench run -c config/smoke.yaml \
  --backend vllm --model-path "$LLMBENCH_MODEL_PATH" \
  --scenarios single --set scenarios.prompt_lengths=[256] \
  --set scenarios.output_lengths=[32] --set scenarios.repeats=1
```

**通过标准**

- [ ] 不抛异常，输出 `✓ Run saved to ...`
- [ ] `llmbench show <run-id>` 中 `ttft_ms_mean` 明显小于 `e2e_ms_mean`
- [ ] 记录中 `extra_engine_api` 为 `v1` 或 `legacy`（说明走了流式路径）
- [ ] 若走了 `offline` 回退，日志中应有 `Streaming engine unavailable` 警告——
      此时需记录实际 vLLM 版本，后续单独适配

对 HF 后端重复一次：

```bash
llmbench run -c config/smoke.yaml \
  --backend naive --model-path "$LLMBENCH_MODEL_PATH" \
  --scenarios single --set scenarios.prompt_lengths=[256] \
  --set scenarios.output_lengths=[32]
```

- [ ] `ttft_ms` 非空且随 prompt 长度单调增长
- [ ] `tpot_ms` 落在合理区间（7B/3090 约 20–30 ms）

### 3.4 完整对比

```bash
bash scripts/run_gpu.sh "$LLMBENCH_MODEL_PATH"     # HF + vLLM 各一轮
LLMBENCH_RESULTS_DIR=data/results llmbench report --all --open
```

**通过标准（与原 v0.1 报告同量级即可，不要求逐位一致）**

| 指标 | 预期区间（3090 / 7B / FP16） | 来源 |
|---|---|---|
| 单请求吞吐（vLLM） | 40–60 tok/s | v0.1 实测 47.3 |
| 并发 8 吞吐（vLLM） | 250–380 tok/s | v0.1 实测 321 |
| 并发 8 吞吐（HF 静态批） | 150–250 tok/s | v0.1 实测 200 |
| 2048 token 的 KV 增量（vLLM） | 300–500 MB | v0.1 实测 388 |
| KV 节省比例 | 70–85% | v0.1 实测 79.8% |
| 前缀缓存 TTFT 加速 | 1.3–2.5× | v0.1 实测 1.75× |

**若超出区间**：先查 `metadata.system`（驱动/版本）、再查 `prompt_len` 是否用了
后端回报的真实 token 数（不是估算值），最后看 `queue_ms` 是否异常。

### 3.5 已知会踩的坑

| 现象 | 原因 | 处理 |
|---|---|---|
| `ValueError: max_model_len` 超限 | 24GB 卡上 `gpu_memory_utilization=0.85` 装不下 8192 | 降到 4096，或调低 `gpu_memory_utilization` |
| 第二个引擎启动时 OOM | 前一个引擎显存未释放 | 已由 `release_gpu()` + `sleep` 处理；若仍失败，把两个后端拆成两次进程运行 |
| `AsyncLLMEngine` 导入失败 | vLLM ≥0.7 移除该路径 | 代码已回退到 v1 的 `AsyncLLM`；两者都失败才降级 offline |
| 中文 prompt 的 token 数对不上 | 用 `len/1.8` 估算 | 记录里的 `prompt_len` 取自后端回报的 `input_tokens`，图表以它为准，估算只用于生成 |

### 3.6 验收后要更新什么

1. `docs/gpu-validation-<date>.md`：新建，附 `llmbench info` 输出、四张关键图、与上表对照。
2. `README.md` 「核心结论」：新增「实测」列。
3. `CHANGELOG.md`：在 `[0.2.0]` 下补一行「Verified on ...」。
4. 若发现 API 不兼容：在 `docs/decisions.md` 追加 `D-16`，而不是直接改代码不留痕。

---

## 4. 发布与协作流程

### 4.1 分支策略

```bash
# 当前状态：v0.2 已在本地 main 上，尚未推送
git push origin main
git tag -a v0.2.0 -m "v0.2.0: auditable benchmark harness"
git push origin v0.2.0
```

若希望走评审流程（推荐，便于日后回看）：

```bash
git checkout -b release/0.2
git push -u origin release/0.2
gh pr create --title "v0.2: rebuild as an auditable benchmark harness" \
             --body-file CHANGELOG.md
```

**为什么推荐 PR 而不是直接 push**：v0.2 是破坏性变更，PR 的 diff 与讨论是
「为什么这么改」最好的存档，和 `docs/decisions.md` 互为印证。

### 4.2 标签

| 标签 | 指向 | 用途 |
|---|---|---|
| `v0.1.0` | `7b047a2` | 审计文档的所有 `文件:行号` 引用基线 |
| `v0.2.0` | 本次重构 | 可运行的第一个版本（模拟验证） |
| `v0.2.1` | 待定 | GPU 验收后的小修（预计会有） |

**不要移动已发布的标签**——审计文档依赖 `v0.1.0` 定位原始代码。

### 4.3 文档维护约定

| 触发条件 | 必须更新 |
|---|---|
| 新增/删除一个架构决策 | `docs/decisions.md` 追加条目（不要改写历史条目） |
| 发现新的 v0.1 缺陷 | `docs/audit-v0.1.md` 追加并标注「审计后补」 |
| 完成一项 roadmap 条目 | `docs/roadmap.md` 移到「已完成」，并在 `CHANGELOG.md` 记一行 |
| 改了测量口径 | `docs/methodology.md` + 相关单测 + `README` 的数字来源说明 |
| 改了 CLI 或配置项 | `README.md`「命令一览」/「配置」+ `--help` 文本 |
| 发布版本 | `CHANGELOG.md` 新增小节，`pyproject.toml` 改 version |

---

## 5. 后续优化（按价值/成本排序）

### P1 — 下一步就该做

| 项 | 价值 | 成本 | 备注 |
|---|---|---|---|
| **GPU 验收**（第 3 节） | 决定项目能否被引用 | 0.5 天 | 唯一阻塞项 |
| vLLM 版本兼容矩阵 | 避免「装完就报错」 | 1 天 | 用 `tox`/矩阵 CI + 三个版本的真实 import 测试 |
| `llmbench compare A B` 子命令 | 目前只能整表对比，缺「两次运行之间的 delta」 | 0.5 天 | 输出 Markdown 表格 + 变更百分比 |

### P2 — 有明显收益

| 项 | 价值 | 备注 |
|---|---|---|
| OpenAI 兼容服务端模式 | 最贴近生产：`vllm serve` + HTTP 流式，TTFT 含网络栈 | 新增 `serving_mode=openai_server`，需子进程管理 |
| 多进程压测 | 绕开 GIL，支持 >100 并发 | `--workers N`，结果按进程合并 |
| 采样精度自适应 | 目前固定 100ms，短请求峰值会漏 | 按 e2e 自动调间隔，或改用 CUDA event |
| 结果对比视图（仪表盘） | 选两个 run 并排看差异 | Streamlit 已能多选，缺「差异」页签 |
| 误差棒 / 置信区间 | `repeats>1` 时应标注方差 | `Analysis` 已有 `n_runs`，缺渲染 |
| 成本模型（$/1M tokens） | 让吞吐对比有商业含义 | 按 GPU 时价换算，需用户填单价 |

### P3 — 锦上添花

| 项 | 备注 |
|---|---|
| chunked prefill 建模 | 模拟器已在 prefill 阶段可扩展，加一个 `max_num_batched_tokens` 即可 |
| 投机解码模拟 | 需要草稿模型命中率参数，收益模型不确定 |
| 张量并行通信开销 | 需要多卡环境标定，单卡无法验证 |
| Prometheus 指标导出 | 若要做「持续压测」才有意义 |
| PyPI 发布 | 当被第三方依赖时再做（见 [D-15](decisions.md#d-15-暂不拆分为独立仓库)） |

### 明确不做

- **不做 Web 后端服务**：Streamlit 已覆盖「看结果」的需求，加 API 只会增加部署负担。
- **不做自动调参**：benchmark 的职责是测量，不是优化；调参交回给 vLLM 自己的参数。
- **不把模拟结果写成实测**：任何时候都不允许去掉 `is_mock` 标记。

---

## 6. 风险登记

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| vLLM 流式 API 与预期不符 | 中 | 高（TTFT 退化） | 三段回退 + 明确日志 + 验收清单第 3.3 节 |
| 模拟结果被误引用 | 中 | 高（学术诚信） | 元数据标记 + 报告横幅 + 文档声明 |
| 24GB 卡上 OOM | 中 | 中 | `max_model_len` 与 `gpu_memory_utilization` 可配；验收清单已列 |
| 线程模型在 >100 并发时失真 | 高 | 中 | methodology 已声明；roadmap P2 |
| 文档与代码漂移 | 中 | 中 | 4.3 节的维护约定 + CI 不校验文档（需人工守） |
| 依赖版本漂移导致 CI 红 | 中 | 低 | requirements 只给下限，CI 每周跑一次即可发现 |

---

**相关文档**：[审计](audit-v0.1.md) · [决策记录](decisions.md) · [方法学](methodology.md) · [架构](architecture.md)
