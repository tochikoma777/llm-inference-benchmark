"""Static (matplotlib) figures for papers, slides and the README.

Plotly powers the interactive HTML report; these exports exist because
publications, slide decks and GitHub previews need a raster/svg file.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..core.logging import get_logger
from ..reporting.analysis import Analysis
from .theme import (
    PALETTE,
    apply_style,
    backend_color,
    label,
    policy_color,
    prime_series_colors,
    save_figure,
)

__all__ = ["make_all_figures"]

log = get_logger("viz.figures")


def _set_log_concurrency_axis(ax, frame: pd.DataFrame) -> None:
    """Log x-axis showing only the measured levels (no 3×10⁰ clutter)."""
    from matplotlib.ticker import FixedLocator, NullFormatter

    levels = [int(v) for v in sorted(frame["concurrency"].dropna().unique())]
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(levels))
    ax.xaxis.set_major_formatter(plt.ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticklabels([str(level) for level in levels])
    ax.set_xlim(min(levels) * 0.85, max(levels) * 1.18)


def _style_axis(ax, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="both", alpha=0.9)
    ax.set_axisbelow(True)


def _fig_throughput(analysis: Analysis, lang: str, outdir: Path) -> list[str] | None:
    frame = analysis.concurrency_frame()
    if frame.empty or "throughput_tok_s" not in frame.columns:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for idx, backend in enumerate(sorted(frame["series"].dropna().unique())):
        subset = frame[frame["series"] == backend].sort_values("concurrency")
        ax.plot(
            subset["concurrency"],
            subset["throughput_tok_s"],
            marker="o",
            label=backend,
            color=backend_color(backend, idx),
        )
    _set_log_concurrency_axis(ax, frame)
    ax.set_ylim(bottom=0)
    _style_axis(
        ax,
        label("concurrency", lang),
        label("throughput", lang),
        "Throughput scaling under concurrency" if lang == "en" else "并发吞吐扩展",
    )
    ax.legend(frameon=False)
    return save_figure(fig, outdir / "01_throughput_vs_concurrency")


def _fig_latency(analysis: Analysis, lang: str, outdir: Path) -> list[str] | None:
    frame = analysis.concurrency_frame()
    if frame.empty or "ttft_ms_p95" not in frame.columns:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for idx, backend in enumerate(sorted(frame["series"].dropna().unique())):
        subset = frame[frame["series"] == backend].sort_values("concurrency")
        color = backend_color(backend, idx)
        ax.plot(
            subset["concurrency"],
            subset["ttft_ms_p95"],
            marker="o",
            color=color,
            label=f"{backend} TTFT p95",
        )
        if "e2e_ms_p99" in subset:
            ax.plot(
                subset["concurrency"],
                subset["e2e_ms_p99"],
                marker="s",
                linestyle="--",
                color=color,
                alpha=0.85,
                label=f"{backend} E2E p99",
            )
    _set_log_concurrency_axis(ax, frame)
    ax.set_yscale("log")
    _style_axis(
        ax,
        label("concurrency", lang),
        label("latency", lang),
        "Tail latency under load" if lang == "en" else "负载下的尾延迟",
    )
    ax.legend(
        frameon=False,
        fontsize=8.5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
    )
    return save_figure(fig, outdir / "02_latency_vs_concurrency")


def _fig_ttft_prompt(analysis: Analysis, lang: str, outdir: Path) -> list[str] | None:
    frame = analysis.latency_frame()
    if frame.empty or "ttft_ms_mean" not in frame.columns:
        return None
    frame = frame.dropna(subset=["prompt_len"])
    if frame.empty:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for idx, backend in enumerate(sorted(frame["series"].dropna().unique())):
        subset = frame[frame["series"] == backend]
        if "requested_output_len" in subset and subset["requested_output_len"].notna().any():
            subset = subset[subset["requested_output_len"] == subset["requested_output_len"].min()]
        subset = subset.sort_values("prompt_len")
        ax.plot(
            subset["prompt_len"],
            subset["ttft_ms_mean"],
            marker="o",
            color=backend_color(backend, idx),
            label=backend,
        )
    _style_axis(
        ax,
        label("prompt_len", lang),
        label("ttft", lang),
        (
            "Prefill cost grows linearly with prompt length"
            if lang == "en"
            else "预填充成本随输入长度线性增长"
        ),
    )
    ax.legend(frameon=False)
    return save_figure(fig, outdir / "03_ttft_vs_prompt_length")


def _fig_memory(analysis: Analysis, lang: str, outdir: Path) -> list[str] | None:
    frame = analysis.memory_frame()
    if frame.empty:
        return None
    metric = "kv_cache_delta_mb" if "kv_cache_delta_mb" in frame.columns else "peak_memory_mb"
    frame = frame.dropna(subset=[metric, "prompt_len"])
    if frame.empty:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for idx, backend in enumerate(sorted(frame["series"].dropna().unique())):
        subset = frame[frame["series"] == backend].sort_values("prompt_len")
        color = backend_color(backend, idx)
        ax.plot(subset["prompt_len"], subset[metric], marker="o", color=color, label=backend)
    _style_axis(
        ax,
        label("prompt_len", lang),
        label("kv_delta", lang) if metric == "kv_cache_delta_mb" else label("memory", lang),
        "KV-cache footprint vs. sequence length" if lang == "en" else "KV Cache 占用随序列长度变化",
    )
    ax.legend(frameon=False)
    return save_figure(fig, outdir / "04_memory_vs_sequence_length")


def _fig_gpu_util(analysis: Analysis, lang: str, outdir: Path) -> list[str] | None:
    frame = analysis.concurrency_frame()
    if frame.empty or "avg_gpu_util_pct" not in frame.columns:
        return None
    frame = frame.dropna(subset=["avg_gpu_util_pct", "concurrency"])
    if frame.empty:
        return None
    backends = sorted(frame["series"].dropna().unique())
    levels = sorted(frame["concurrency"].dropna().unique())
    width = 0.8 / max(len(backends), 1)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(levels))
    for idx, backend in enumerate(backends):
        subset = frame[frame["series"] == backend].set_index("concurrency")
        values = [subset["avg_gpu_util_pct"].get(level, np.nan) for level in levels]
        ax.bar(x + idx * width, values, width, label=backend, color=backend_color(backend, idx))
    ax.set_xticks(x + width * (len(backends) - 1) / 2)
    ax.set_xticklabels([str(int(level)) for level in levels])
    ax.set_ylim(0, 100)
    _style_axis(
        ax,
        label("concurrency", lang),
        label("gpu_util", lang),
        "GPU utilisation: idle vs. saturated" if lang == "en" else "GPU 利用率对比",
    )
    ax.legend(frameon=False)
    return save_figure(fig, outdir / "05_gpu_utilization")


def _fig_prefix(analysis: Analysis, lang: str, outdir: Path) -> list[str] | None:
    frame = analysis.prefix_frame()
    if frame.empty or "ttft_ms_mean" not in frame.columns:
        return None
    frame = frame.dropna(subset=["ttft_ms_mean"]).copy()
    if frame.empty:
        return None
    frame["round"] = frame["group"].astype(str).str.extract(r"_r(\d+)").astype(float)
    frame = frame.sort_values("round")
    groups = sorted(frame["group"].astype(str).str.replace(r"_r\d+$", "", regex=True).unique())
    levels = sorted(int(v) for v in frame["round"].dropna().unique())
    width = 0.8 / max(len(groups), 1)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(levels))
    for idx, group in enumerate(groups):
        subset = frame[frame["group"].astype(str).str.startswith(group)].set_index("round")
        values = [subset["ttft_ms_mean"].get(float(level), np.nan) for level in levels]
        ax.bar(
            x + idx * width,
            values,
            width,
            label="cache off" if group == "cache_off" else "cache on",
            color=PALETTE["rose"] if group == "cache_off" else PALETTE["emerald"],
        )
    ax.set_xticks(x + width * (len(groups) - 1) / 2)
    ax.set_xticklabels([str(level) for level in levels])
    _style_axis(
        ax,
        label("round", lang),
        label("ttft", lang),
        "Prefix caching across conversation turns" if lang == "en" else "多轮对话中的前缀缓存收益",
    )
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=len(groups))
    return save_figure(fig, outdir / "06_prefix_caching")


def policy_label(row) -> str:
    """Robust policy label: explicit label → serving_mode → '?'."""
    text = row.get("extra_policy_label")
    if isinstance(text, str) and text.strip():
        return text
    mode = row.get("serving_mode")
    return str(mode) if isinstance(mode, str) and mode else "?"


def _fig_scheduler(analysis: Analysis, lang: str, outdir: Path) -> list[str] | None:
    frame = analysis.scheduler_frame()
    if frame.empty or "ttft_ms_mean" not in frame.columns:
        return None
    frame = frame.dropna(subset=["ttft_ms_mean"]).sort_values("ttft_ms_mean").reset_index(drop=True)
    if frame.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    labels = [policy_label(row) for _, row in frame.iterrows()]
    colors = [policy_color(str(p), i) for i, p in enumerate(frame["serving_mode"])]
    x = np.arange(len(labels))

    axes[0].bar(x, frame["ttft_ms_mean"], color=colors, width=0.62)
    for xi, value in zip(x, frame["ttft_ms_mean"], strict=False):
        axes[0].text(
            xi,
            value,
            f" {value:,.0f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=PALETTE["slate"],
        )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=15, ha="right")
    _style_axis(
        axes[0],
        label("policy", lang),
        label("ttft", lang),
        "Mean TTFT" if lang == "en" else "平均首 Token 延迟",
    )

    if {"extra_short_ttft_ms", "extra_long_ttft_ms"} <= set(frame.columns):
        width = 0.38
        axes[1].bar(
            x - width / 2,
            frame["extra_short_ttft_ms"],
            width,
            label="short",
            color=PALETTE["emerald"],
        )
        axes[1].bar(
            x + width / 2, frame["extra_long_ttft_ms"], width, label="long", color=PALETTE["rose"]
        )
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels, rotation=15, ha="right")
        _style_axis(
            axes[1],
            label("policy", lang),
            label("ttft", lang),
            "Fairness split" if lang == "en" else "公平性对比",
        )
        axes[1].legend(
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.34),
            ncol=2,
        )

    fig.tight_layout()
    return save_figure(fig, outdir / "07_scheduler_comparison")


def _fig_cdf(analysis: Analysis, lang: str, outdir: Path) -> list[str] | None:
    raw = analysis.raw
    if raw.empty or "e2e_ms" not in raw.columns:
        return None
    conc = raw[raw.get("scenario") == "concurrency"].dropna(subset=["e2e_ms", "concurrency"])
    if conc.empty:
        return None
    top = conc["concurrency"].max()
    conc = conc[conc["concurrency"] == top]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for idx, backend in enumerate(conc["series"].dropna().unique()):
        values = np.sort(conc[conc["series"] == backend]["e2e_ms"].to_numpy())
        if values.size == 0:
            continue
        cdf = np.arange(1, values.size + 1) / values.size * 100.0
        ax.plot(values, cdf, color=backend_color(backend, idx), label=backend)
    ax.axhline(95, color=PALETTE["muted"], linestyle=":", linewidth=1)
    _style_axis(
        ax,
        label("e2e", lang),
        label("cdf", lang),
        (
            f"Latency distribution at concurrency={int(top)}"
            if lang == "en"
            else f"端到端延迟分布（并发={int(top)}）"
        ),
    )
    ax.legend(frameon=False)
    return save_figure(fig, outdir / "08_latency_cdf")


_BUILDERS = (
    _fig_throughput,
    _fig_latency,
    _fig_ttft_prompt,
    _fig_memory,
    _fig_gpu_util,
    _fig_prefix,
    _fig_scheduler,
    _fig_cdf,
)


def make_all_figures(
    analysis: Analysis,
    outdir: Path | None = None,
    lang: str = "auto",
    formats: tuple[str, ...] = ("png",),
) -> list[str]:
    """Render every applicable figure and return the written file paths."""
    from ..core.paths import figures_dir

    target = Path(outdir) if outdir else figures_dir()
    target.mkdir(parents=True, exist_ok=True)
    resolved_lang = apply_style(lang)
    prime_series_colors(analysis.series())

    written: list[str] = []
    for builder in _BUILDERS:
        try:
            paths = builder(analysis, resolved_lang, target)
        except Exception as exc:  # noqa: BLE001 - one bad chart must not kill the report
            log.warning("Figure %s failed: %s", builder.__name__, exc)
            continue
        if paths:
            written.extend(paths)
    log.info("Rendered %d figure file(s) into %s", len(written), target)
    return written
