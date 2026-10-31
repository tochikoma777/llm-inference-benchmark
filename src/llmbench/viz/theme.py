"""Shared visual identity: colours, matplotlib style, bilingual labels.

Two rendering paths share this module:

* **Plotly** (HTML report + Streamlit) — rendered by the browser, so CJK text
  is always safe.
* **matplotlib** (PNG/SVG exports for papers and slides) — CJK is only used
  when a CJK font is actually installed, otherwise labels fall back to
  English instead of rendering tofu boxes.
"""

from __future__ import annotations

import functools

import matplotlib

matplotlib.use("Agg")  # headless-safe; must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

__all__ = [
    "PALETTE",
    "prime_series_colors",
    "BACKEND_COLORS",
    "POLICY_COLORS",
    "SCENARIO_LABELS",
    "has_cjk_font",
    "resolve_language",
    "label",
    "apply_style",
    "backend_color",
    "policy_color",
    "save_figure",
    "format_tokens",
    "format_ms",
]

# ---------------------------------------------------------------------------
# Palette — chosen for colour-blind safety and print contrast.
# ---------------------------------------------------------------------------
PALETTE: dict[str, str] = {
    "blue": "#2563EB",
    "amber": "#F59E0B",
    "violet": "#8B5CF6",
    "emerald": "#10B981",
    "rose": "#EF4444",
    "slate": "#64748B",
    "ink": "#0F172A",
    "muted": "#94A3B8",
    "grid": "#E2E8F0",
    "surface": "#F8FAFC",
}

BACKEND_COLORS: dict[str, str] = {
    "vllm": PALETTE["blue"],
    "naive": PALETTE["amber"],
    "mock": PALETTE["violet"],
    "simulation": PALETTE["emerald"],
}

POLICY_COLORS: dict[str, str] = {
    "fcfs": "#2563EB",
    "sjf": "#F59E0B",
    "priority": "#10B981",
    "sjf_aging": "#8B5CF6",
    "round_robin": "#64748B",
}

SCENARIO_LABELS: dict[str, str] = {
    "single": "Latency sweep",
    "concurrency": "Concurrency",
    "memory": "Memory",
    "prefix_cache": "Prefix cache",
    "scheduler": "Scheduler",
}

_CJK_CANDIDATES = (
    "Noto Sans CJK SC",
    "Noto Sans SC",
    "Source Han Sans SC",
    "WenQuanYi Zen Hei",
    "WenQuanYi Micro Hei",
    "Microsoft YaHei",
    "SimHei",
    "PingFang SC",
    "Heiti SC",
    "Hiragino Sans GB",
)


@functools.lru_cache(maxsize=1)
def _installed_fonts() -> set[str]:
    try:
        return {f.name for f in font_manager.fontManager.ttflist}
    except Exception:  # pragma: no cover
        return set()


@functools.lru_cache(maxsize=1)
def has_cjk_font() -> bool:
    installed = _installed_fonts()
    return any(name in installed for name in _CJK_CANDIDATES)


def _cjk_font_name() -> str | None:
    installed = _installed_fonts()
    for name in _CJK_CANDIDATES:
        if name in installed:
            return name
    return None


def resolve_language(requested: str = "auto") -> str:
    """Resolve ``auto|en|zh`` to a language matplotlib can actually render."""
    if requested == "zh":
        return "zh" if has_cjk_font() else "en"
    if requested == "en":
        return "en"
    return "zh" if has_cjk_font() else "en"


# ---------------------------------------------------------------------------
# Bilingual label table
# ---------------------------------------------------------------------------
_LABELS: dict[str, dict[str, str]] = {
    "ttft": {"en": "TTFT (ms)", "zh": "首 Token 延迟 (ms)"},
    "tpot": {"en": "TPOT (ms/token)", "zh": "单 Token 延迟 (ms/token)"},
    "e2e": {"en": "End-to-end latency (ms)", "zh": "端到端延迟 (ms)"},
    "throughput": {"en": "Throughput (tok/s)", "zh": "吞吐 (tok/s)"},
    "concurrency": {"en": "Concurrency", "zh": "并发数"},
    "prompt_len": {"en": "Prompt length (tokens)", "zh": "输入长度 (tokens)"},
    "output_len": {"en": "Output length (tokens)", "zh": "输出长度 (tokens)"},
    "memory": {"en": "Peak GPU memory (MB)", "zh": "显存峰值 (MB)"},
    "kv_delta": {"en": "KV cache growth (MB)", "zh": "KV Cache 增量 (MB)"},
    "gpu_util": {"en": "GPU utilization (%)", "zh": "GPU 利用率 (%)"},
    "latency": {"en": "Latency (ms)", "zh": "延迟 (ms)"},
    "policy": {"en": "Scheduling policy", "zh": "调度策略"},
    "round": {"en": "Conversation round", "zh": "对话轮次"},
    "speedup": {"en": "Speed-up (×)", "zh": "加速比 (×)"},
    "completion": {"en": "Completion time (s)", "zh": "完成时间 (s)"},
    "queue": {"en": "Queue wait (ms)", "zh": "排队等待 (ms)"},
    "tokens_per_s": {"en": "tokens/s", "zh": "tokens/s"},
    "cdf": {"en": "Cumulative fraction", "zh": "累计占比"},
    "request": {"en": "Request #", "zh": "请求 #"},
    "time": {"en": "Time (s)", "zh": "时间 (s)"},
    "efficiency": {"en": "Throughput per GB (tok/s/GB)", "zh": "每 GB 显存吞吐 (tok/s/GB)"},
}


def label(key: str, lang: str = "en") -> str:
    """Look up a translated axis label, falling back to the key itself."""
    entry = _LABELS.get(key)
    if not entry:
        return key
    return entry.get(lang) or entry["en"]


def prime_series_colors(names) -> None:
    """Assign colours deterministically before rendering any chart.

    Without this, the colour of a series would depend on the order in which
    charts happened to touch it, so the same backend could be blue in one
    figure and amber in the next.
    """
    _SERIES_COLORS.clear()
    for name in sorted({str(n) for n in names if n}):
        backend_color(name)


def backend_color(name: str | None, index: int = 0) -> str:
    """Stable colour per *series* label (backend, or backend · serving mode).

    Colours are memoised so the same series keeps its colour across every
    chart in a report, and so two runs of the same engine (e.g. mock in
    continuous vs. static mode) never collide.
    """
    key = str(name) if name is not None else "?"
    cached = _SERIES_COLORS.get(key)
    if cached:
        return cached

    low = key.lower()
    used = set(_SERIES_COLORS.values())
    for needle, color in _SEMANTIC_COLORS:
        if needle in low and color not in used:
            _SERIES_COLORS[key] = color
            return color
    for color in _SERIES_PALETTE:
        if color not in used:
            _SERIES_COLORS[key] = color
            return color
    fallback = _SERIES_PALETTE[len(_SERIES_COLORS) % len(_SERIES_PALETTE)]
    _SERIES_COLORS[key] = fallback
    return fallback


#: Semantic colours: batching strategy first, engine name second.
_SEMANTIC_COLORS = (
    ("continuous", "#2563EB"),
    ("vllm", "#2563EB"),
    ("static", "#F59E0B"),
    ("naive", "#F59E0B"),
    ("per_request", "#8B5CF6"),
    ("simulation", "#10B981"),
    ("mock", "#8B5CF6"),
)

#: Extended palette used when several series share the same backend.
_SERIES_PALETTE = [
    "#2563EB",
    "#F59E0B",
    "#8B5CF6",
    "#10B981",
    "#EF4444",
    "#0EA5E9",
    "#84CC16",
    "#F472B6",
    "#14B8A6",
    "#A855F7",
]

_SERIES_COLORS: dict[str, str] = {}


def policy_color(name: str | None, index: int = 0) -> str:
    if name in POLICY_COLORS:
        return POLICY_COLORS[name]
    return backend_color(name, index)


# ---------------------------------------------------------------------------
# matplotlib style
# ---------------------------------------------------------------------------
def apply_style(lang: str = "auto") -> str:
    """Apply the house style and return the resolved language."""
    resolved = resolve_language(lang)
    family = ["DejaVu Sans"]
    if resolved == "zh":
        cjk = _cjk_font_name()
        if cjk:
            family = [cjk, "DejaVu Sans"]

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            "font.sans-serif": family,
            "font.size": 10.5,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.titlepad": 14,
            "axes.labelsize": 10.5,
            "axes.labelcolor": "#334155",
            "axes.edgecolor": PALETTE["grid"],
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.8,
            "grid.alpha": 1.0,
            "xtick.color": "#64748B",
            "ytick.color": "#64748B",
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.frameon": False,
            "legend.fontsize": 9.5,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.unicode_minus": False,
        }
    )
    return resolved


def save_figure(fig, path, formats: tuple[str, ...] = ("png",)) -> list[str]:
    """Save a matplotlib figure in several formats; return the written paths."""
    from pathlib import Path

    out: list[str] = []
    base = Path(path)
    for fmt in formats:
        target = base.with_suffix(f".{fmt}")
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, format=fmt)
        out.append(str(target))
    plt.close(fig)
    return out


def format_tokens(value: float | None) -> str:
    if value is None:
        return "—"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}k"
    return f"{value:.0f}"


def format_ms(value: float | None) -> str:
    if value is None:
        return "—"
    if value >= 1000:
        return f"{value / 1000:.2f}s"
    return f"{value:.1f}ms"
