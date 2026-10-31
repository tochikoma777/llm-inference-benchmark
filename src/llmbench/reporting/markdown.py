"""Markdown report — the version that renders on GitHub.

The HTML report is for humans who want charts; this one is for pull requests,
`docs/` pages and anyone reading the raw file in an editor.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..core.paths import reports_dir
from ..reporting.analysis import Analysis
from ..viz.theme import format_ms, format_tokens

__all__ = ["build_markdown_report", "dataframe_to_markdown"]


def dataframe_to_markdown(frame: pd.DataFrame, floatfmt: str = "%.2f") -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table."""
    if frame is None or frame.empty:
        return "_No data_\n"
    columns = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join([":---"] * len(columns)) + "|"]
    for _, row in frame.iterrows():
        cells: list[str] = []
        for value in row:
            if value is None or (isinstance(value, float) and pd.isna(value)):
                cells.append("—")
            elif isinstance(value, float) and float(value).is_integer():
                cells.append(f"{value:,.0f}")
            elif isinstance(value, float):
                cells.append(floatfmt % value)
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def build_markdown_report(analysis: Analysis, out_path: Path | None = None) -> Path:
    out_path = Path(out_path) if out_path else reports_dir() / "benchmark_report.generated.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kpis = analysis.kpis()
    meta = analysis.runs[0].metadata if analysis.runs else None
    lines: list[str] = []

    lines.append("# LLM Inference Benchmark — 自动生成报告\n")
    if analysis.is_mock:
        lines.append(
            "> ⚠️ **模拟数据**：本报告由 mock 后端生成，仅用于验证流水线，不可作为硬件实测结论。\n"
        )
    if meta:
        lines.append(
            f"- **运行**: `{meta.run_id}`\n"
            f"- **后端**: `{meta.backend}` ({meta.serving_mode})\n"
            f"- **模型**: `{meta.model_id or '—'}`\n"
            f"- **GPU**: {(meta.system or {}).get('gpu_name') or '—'}\n"
            f"- **时间**: {meta.created_at}\n"
        )

    lines.append("\n## 关键指标\n")
    rows = []
    if kpis.get("peak_throughput_tok_s"):
        rows.append(
            (
                "峰值吞吐",
                f"{kpis['peak_throughput_tok_s']:,.0f} tok/s",
                f"{kpis.get('peak_throughput_backend')} @ c={kpis.get('peak_concurrency')}",
            )
        )
    if kpis.get("throughput_speedup"):
        rows.append(
            (
                "吞吐加速比",
                f"{kpis['throughput_speedup']:.2f}×",
                f"{kpis.get('speedup_fastest','')} vs {kpis.get('speedup_baseline','')}",
            )
        )
    if kpis.get("best_ttft_p95_ms") is not None:
        rows.append(("最优 TTFT p95", format_ms(kpis["best_ttft_p95_ms"]), "高并发下"))
    if kpis.get("memory_saving_pct") is not None:
        rows.append(
            (
                "KV Cache 节省",
                f"{kpis['memory_saving_pct']:.0f}%",
                f"{kpis.get('memory_best_backend')} @ {format_tokens(kpis.get('kv_delta_at_tokens'))} tokens",
            )
        )
    if kpis.get("prefix_speedup"):
        rows.append(("前缀缓存加速", f"{kpis['prefix_speedup']:.2f}×", "多轮对话末轮 TTFT"))
    if kpis.get("best_policy"):
        rows.append(
            (
                "最优调度策略",
                str(kpis["best_policy"]).upper(),
                f"平均 TTFT {format_ms(kpis.get('best_policy_ttft_ms'))}",
            )
        )
    if rows:
        lines.append("| 指标 | 数值 | 说明 |")
        lines.append("|:---|:---|:---|")
        for name, value, hint in rows:
            lines.append(f"| {name} | **{value}** | {hint} |")
    else:
        lines.append("_无可用指标_\n")

    comparison = analysis.comparison()
    if not comparison.empty:
        lines.append("\n## 后端对比（最高并发）\n")
        lines.append(dataframe_to_markdown(comparison))

    conc = analysis.concurrency_frame()
    if not conc.empty:
        lines.append("\n## 并发扩展\n")
        columns = [
            c
            for c in (
                "series",
                "concurrency",
                "throughput_tok_s",
                "ttft_ms_p95",
                "e2e_ms_p99",
                "avg_gpu_util_pct",
            )
            if c in conc.columns
        ]
        view = conc[columns].rename(
            columns={
                "series": "后端",
                "concurrency": "并发",
                "throughput_tok_s": "吞吐 (tok/s)",
                "ttft_ms_p95": "TTFT p95 (ms)",
                "e2e_ms_p99": "E2E p99 (ms)",
                "avg_gpu_util_pct": "GPU 利用率 (%)",
            }
        )
        lines.append(dataframe_to_markdown(view))

    prefix = analysis.prefix_frame()
    if not prefix.empty:
        lines.append("\n## 前缀缓存（多轮对话）\n")
        columns = [
            c for c in ("group", "prompt_len", "ttft_ms_mean", "e2e_ms_mean") if c in prefix.columns
        ]
        lines.append(
            dataframe_to_markdown(
                prefix[columns].rename(
                    columns={
                        "group": "分组",
                        "prompt_len": "输入长度",
                        "ttft_ms_mean": "TTFT (ms)",
                        "e2e_ms_mean": "端到端 (ms)",
                    }
                )
            )
        )

    sched = analysis.scheduler_frame()
    if not sched.empty:
        lines.append("\n## 调度策略对比\n")
        columns = [
            c
            for c in (
                "extra_policy_label",
                "ttft_ms_mean",
                "ttft_ms_p95",
                "e2e_ms_p99",
                "throughput_tok_s",
                "queue_ms_mean",
                "extra_preemptions",
                "extra_fairness_jain",
            )
            if c in sched.columns
        ]
        view = sched[columns].rename(
            columns={
                "extra_policy_label": "策略",
                "ttft_ms_mean": "平均 TTFT (ms)",
                "ttft_ms_p95": "TTFT p95 (ms)",
                "e2e_ms_p99": "E2E p99 (ms)",
                "throughput_tok_s": "吞吐 (tok/s)",
                "queue_ms_mean": "平均排队 (ms)",
                "extra_preemptions": "抢占次数",
                "extra_fairness_jain": "公平性 (Jain)",
            }
        )
        lines.append(dataframe_to_markdown(view))

    lines.append("\n## 复现\n")
    lines.append(
        "```bash\nllmbench run -c config/vllm.yaml --model-path /path/to/model\n"
        "llmbench report --latest\n```\n"
    )

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path
