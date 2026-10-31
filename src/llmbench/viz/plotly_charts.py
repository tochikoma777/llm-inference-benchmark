"""Plotly chart builders shared by the HTML report and the Streamlit app.

Each builder accepts an :class:`~llmbench.reporting.analysis.Analysis` (or a
pre-filtered frame) and returns a ``plotly.graph_objects.Figure`` — or ``None``
when the required data is absent, so callers can skip empty sections gracefully.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from ..reporting.analysis import Analysis
from .theme import PALETTE, backend_color, policy_color

__all__ = [
    "FONT_FAMILY",
    "throughput_vs_concurrency",
    "throughput_speedup",
    "latency_vs_concurrency",
    "ttft_vs_prompt_length",
    "tpot_vs_prompt_length",
    "memory_vs_sequence_length",
    "gpu_utilization",
    "prefix_cache_rounds",
    "scheduler_policy_bars",
    "scheduler_group_split",
    "latency_cdf",
    "request_timeline",
    "efficiency_pareto",
    "empty_figure",
]

FONT_FAMILY = (
    "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans SC', "
    "'PingFang SC', 'Microsoft YaHei', sans-serif"
)


def _base_layout(
    fig: go.Figure,
    title: str,
    x_title: str = "",
    y_title: str = "",
    height: int = 380,
    show_legend: bool = True,
) -> go.Figure:
    fig.update_layout(
        title={
            "text": title,
            "font": {"size": 15, "color": PALETTE["ink"]},
            "x": 0,
            "xanchor": "left",
        },
        font={"family": FONT_FAMILY, "size": 12, "color": "#334155"},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin={"l": 64, "r": 24, "t": 56, "b": 48},
        hovermode="x unified",
        showlegend=show_legend,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "bgcolor": "rgba(0,0,0,0)",
        },
        xaxis={
            "title": {"text": x_title, "font": {"size": 12}},
            "gridcolor": PALETTE["grid"],
            "zeroline": False,
            "linecolor": PALETTE["grid"],
        },
        yaxis={
            "title": {"text": y_title, "font": {"size": 12}},
            "gridcolor": PALETTE["grid"],
            "zeroline": False,
            "linecolor": PALETTE["grid"],
        },
    )
    return fig


def empty_figure(message: str = "No data for this scenario") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 13, "color": PALETTE["muted"], "family": FONT_FAMILY},
    )
    fig.update_layout(
        height=280,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"visible": False},
        yaxis={"visible": False},
        margin={"l": 24, "r": 24, "t": 24, "b": 24},
    )
    return fig


def _series_label(row: pd.Series) -> str:
    label_text = row.get("series")
    if label_text:
        return str(label_text)
    backend = str(row.get("backend") or "?")
    mode = row.get("serving_mode")
    if mode and mode not in {"continuous_batch"}:
        return f"{backend} · {mode}"
    return backend


# ---------------------------------------------------------------------------
# Throughput
# ---------------------------------------------------------------------------
def throughput_vs_concurrency(analysis: Analysis) -> go.Figure | None:
    frame = analysis.concurrency_frame()
    if frame.empty or "throughput_tok_s" not in frame.columns:
        return None

    fig = go.Figure()
    for idx, backend in enumerate(frame["series"].dropna().unique()):
        subset = frame[frame["series"] == backend].sort_values("concurrency")
        color = backend_color(backend, idx)
        fig.add_trace(
            go.Scatter(
                x=subset["concurrency"],
                y=subset["throughput_tok_s"],
                mode="lines+markers",
                name=_series_label(subset.iloc[0]),
                line={"color": color, "width": 2.6},
                marker={"size": 9, "color": color, "line": {"width": 2, "color": "white"}},
                customdata=subset[["ttft_ms_p95"]].to_numpy() if "ttft_ms_p95" in subset else None,
                hovertemplate=(
                    "concurrency=%{x}<br>throughput=%{y:.1f} tok/s"
                    + ("<br>TTFT p95=%{customdata[0]:.0f} ms" if "ttft_ms_p95" in subset else "")
                    + "<extra>%{fullData.name}</extra>"
                ),
            )
        )

    fig.update_xaxes(
        type="log", tickmode="array", tickvals=sorted(frame["concurrency"].dropna().unique())
    )
    return _base_layout(
        fig,
        "Throughput vs. concurrency",
        "Concurrent requests (log scale)",
        "Output tokens / second",
    )


def throughput_speedup(analysis: Analysis) -> go.Figure | None:
    """Relative speed-up of every backend against the slowest one per level."""
    frame = analysis.concurrency_frame()
    if frame.empty or "throughput_tok_s" not in frame.columns:
        return None
    pivot = frame.pivot_table(
        index="concurrency", columns="series", values="throughput_tok_s", aggfunc="max"
    )
    if pivot.empty or pivot.shape[1] < 2:
        return None
    comparable = pivot.dropna()
    if comparable.empty:
        return None
    baseline_name = comparable.mean().idxmin()
    baseline = pivot[baseline_name]

    fig = go.Figure()
    for idx, backend in enumerate(pivot.columns):
        if backend == baseline_name:
            continue
        ratio = pivot[backend] / baseline
        color = backend_color(backend, idx)
        fig.add_trace(
            go.Scatter(
                x=ratio.index,
                y=ratio.values,
                mode="lines+markers",
                name=f"{backend} vs {baseline_name}",
                line={"color": color, "width": 2.6},
                marker={"size": 9, "color": color, "line": {"width": 2, "color": "white"}},
                hovertemplate="concurrency=%{x}<br>speed-up=%{y:.2f}×<extra></extra>",
            )
        )
    fig.add_hline(y=1.0, line_dash="dot", line_color=PALETTE["muted"], annotation_text="parity")
    fig.update_xaxes(type="log", tickmode="array", tickvals=sorted(pivot.index.dropna().unique()))
    return _base_layout(
        fig, "Throughput speed-up over baseline", "Concurrent requests", "Speed-up (×)"
    )


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------
def latency_vs_concurrency(analysis: Analysis) -> go.Figure | None:
    frame = analysis.concurrency_frame()
    if frame.empty:
        return None

    fig = go.Figure()
    for idx, backend in enumerate(frame["series"].dropna().unique()):
        subset = frame[frame["series"] == backend].sort_values("concurrency")
        color = backend_color(backend, idx)
        if "ttft_ms_p95" in subset:
            fig.add_trace(
                go.Scatter(
                    x=subset["concurrency"],
                    y=subset["ttft_ms_p95"],
                    mode="lines+markers",
                    name=f"{backend} · TTFT p95",
                    line={"color": color, "width": 2.6},
                    marker={"size": 8, "color": color},
                    hovertemplate="concurrency=%{x}<br>TTFT p95=%{y:.0f} ms<extra></extra>",
                )
            )
        if "e2e_ms_p99" in subset:
            fig.add_trace(
                go.Scatter(
                    x=subset["concurrency"],
                    y=subset["e2e_ms_p99"],
                    mode="lines+markers",
                    name=f"{backend} · E2E p99",
                    line={"color": color, "width": 2, "dash": "dash"},
                    marker={"size": 7, "symbol": "diamond", "color": color},
                    hovertemplate="concurrency=%{x}<br>E2E p99=%{y:.0f} ms<extra></extra>",
                )
            )
    if not fig.data:
        return None
    fig.update_xaxes(
        type="log", tickmode="array", tickvals=sorted(frame["concurrency"].dropna().unique())
    )
    fig.update_yaxes(type="log")
    return _base_layout(
        fig, "Tail latency vs. concurrency", "Concurrent requests", "Latency (ms, log)"
    )


def ttft_vs_prompt_length(analysis: Analysis) -> go.Figure | None:
    frame = analysis.latency_frame()
    if frame.empty or "ttft_ms_mean" not in frame.columns:
        return None
    frame = frame.dropna(subset=["prompt_len"])
    if frame.empty:
        return None

    fig = go.Figure()
    for idx, backend in enumerate(frame["series"].dropna().unique()):
        subset = frame[frame["series"] == backend]
        # Use the shortest output length so TTFT dominates the measurement.
        if "requested_output_len" in subset:
            smallest = subset["requested_output_len"].min()
            if pd.notna(smallest):
                subset = subset[subset["requested_output_len"] == smallest]
        subset = subset.sort_values("prompt_len")
        color = backend_color(backend, idx)
        fig.add_trace(
            go.Scatter(
                x=subset["prompt_len"],
                y=subset["ttft_ms_mean"],
                mode="lines+markers",
                name=backend,
                line={"color": color, "width": 2.6},
                marker={"size": 9, "color": color, "line": {"width": 2, "color": "white"}},
                hovertemplate="prompt=%{x} tok<br>TTFT=%{y:.0f} ms<extra></extra>",
            )
        )
    return _base_layout(
        fig, "Time to first token vs. prompt length", "Prompt length (tokens)", "TTFT (ms)"
    )


def tpot_vs_prompt_length(analysis: Analysis) -> go.Figure | None:
    frame = analysis.latency_frame()
    if frame.empty or "tpot_ms_mean" not in frame.columns:
        return None
    frame = frame.dropna(subset=["tpot_ms_mean", "prompt_len"])
    if frame.empty:
        return None

    fig = go.Figure()
    for idx, backend in enumerate(frame["series"].dropna().unique()):
        subset = frame[frame["series"] == backend].sort_values("prompt_len")
        color = backend_color(backend, idx)
        fig.add_trace(
            go.Bar(
                x=subset["prompt_len"].astype(str),
                y=subset["tpot_ms_mean"],
                name=backend,
                marker_color=color,
                opacity=0.9,
                hovertemplate="prompt=%{x} tok<br>TPOT=%{y:.2f} ms<extra></extra>",
            )
        )
    fig.update_layout(barmode="group")
    return _base_layout(fig, "Decode cost per token", "Prompt length (tokens)", "TPOT (ms/token)")


def latency_cdf(analysis: Analysis) -> go.Figure | None:
    """CDF of end-to-end latency for request-level records."""
    raw = analysis.raw
    if raw.empty or "e2e_ms" not in raw.columns:
        return None
    conc = raw[raw.get("scenario") == "concurrency"].dropna(subset=["e2e_ms", "concurrency"])
    if conc.empty:
        return None
    top = conc["concurrency"].max()
    conc = conc[conc["concurrency"] == top]

    fig = go.Figure()
    for idx, backend in enumerate(conc["series"].dropna().unique()):
        values = sorted(conc[conc["series"] == backend]["e2e_ms"].tolist())
        if not values:
            continue
        n = len(values)
        cdf = [(i + 1) / n * 100 for i in range(n)]
        color = backend_color(backend, idx)
        fig.add_trace(
            go.Scatter(
                x=values,
                y=cdf,
                mode="lines",
                name=backend,
                line={"color": color, "width": 2.6},
                hovertemplate="latency=%{x:.0f} ms<br>p=%{y:.0f}%<extra></extra>",
            )
        )
        p95 = values[min(int(round(0.95 * (n - 1))), n - 1)]
        fig.add_trace(
            go.Scatter(
                x=[p95],
                y=[95],
                mode="markers",
                showlegend=False,
                marker={"size": 9, "color": color, "symbol": "circle-open", "line": {"width": 3}},
                hovertemplate=f"{backend} p95 = %{{x:.0f}} ms<extra></extra>",
            )
        )
    if not fig.data:
        return None
    return _base_layout(
        fig,
        f"End-to-end latency distribution (concurrency={int(top)})",
        "End-to-end latency (ms)",
        "Cumulative share of requests (%)",
    )


# ---------------------------------------------------------------------------
# Memory / utilisation
# ---------------------------------------------------------------------------
def memory_vs_sequence_length(analysis: Analysis) -> go.Figure | None:
    frame = analysis.memory_frame()
    if frame.empty:
        return None
    metric = "kv_cache_delta_mb" if "kv_cache_delta_mb" in frame.columns else "peak_memory_mb"
    frame = frame.dropna(subset=[metric, "prompt_len"])
    if frame.empty:
        return None

    fig = go.Figure()
    for idx, backend in enumerate(frame["series"].dropna().unique()):
        subset = frame[frame["series"] == backend].sort_values("prompt_len")
        color = backend_color(backend, idx)
        fig.add_trace(
            go.Scatter(
                x=subset["prompt_len"],
                y=subset[metric],
                mode="lines+markers",
                name=backend,
                line={"color": color, "width": 2.6},
                marker={"size": 9, "color": color, "line": {"width": 2, "color": "white"}},
                hovertemplate="tokens=%{x}<br>memory=%{y:.0f} MB<extra></extra>",
            )
        )
    title = (
        "KV-cache growth vs. sequence length"
        if metric == "kv_cache_delta_mb"
        else "Peak GPU memory vs. sequence length"
    )
    y_title = "KV cache growth (MB)" if metric == "kv_cache_delta_mb" else "Peak memory (MB)"
    return _base_layout(fig, title, "Sequence length (tokens)", y_title)


def gpu_utilization(analysis: Analysis) -> go.Figure | None:
    frame = analysis.concurrency_frame()
    if frame.empty or "avg_gpu_util_pct" not in frame.columns:
        return None
    frame = frame.dropna(subset=["avg_gpu_util_pct", "concurrency"])
    if frame.empty:
        return None

    fig = go.Figure()
    for idx, backend in enumerate(frame["series"].dropna().unique()):
        subset = frame[frame["series"] == backend].sort_values("concurrency")
        color = backend_color(backend, idx)
        fig.add_trace(
            go.Bar(
                x=subset["concurrency"].astype(str),
                y=subset["avg_gpu_util_pct"],
                name=backend,
                marker_color=color,
                hovertemplate="concurrency=%{x}<br>GPU util=%{y:.1f}%<extra></extra>",
            )
        )
    fig.update_layout(barmode="group")
    fig.update_yaxes(range=[0, 100])
    return _base_layout(
        fig, "GPU utilisation under load", "Concurrent requests", "Average GPU utilisation (%)"
    )


def prefix_cache_rounds(analysis: Analysis) -> go.Figure | None:
    frame = analysis.prefix_frame()
    if frame.empty or "ttft_ms_mean" not in frame.columns:
        return None
    frame = frame.dropna(subset=["ttft_ms_mean"])
    if frame.empty:
        return None
    frame = frame.copy()
    frame["round"] = frame["group"].astype(str).str.extract(r"_r(\d+)").astype(float)
    frame = frame.sort_values(["group", "round"])

    fig = go.Figure()
    palette = {"cache_off": PALETTE["rose"], "cache_on": PALETTE["emerald"]}
    for name, subset in frame.groupby(
        frame["group"].astype(str).str.replace(r"_r\d+$", "", regex=True)
    ):
        color = palette.get(name, PALETTE["blue"])
        label_text = "cache off" if name == "cache_off" else "cache on"
        fig.add_trace(
            go.Bar(
                x=subset["round"].astype(int).astype(str),
                y=subset["ttft_ms_mean"],
                name=label_text,
                marker_color=color,
                hovertemplate="round %{x}<br>TTFT=%{y:.0f} ms<extra></extra>",
            )
        )
    fig.update_layout(barmode="group")
    return _base_layout(
        fig,
        "Prefix caching on a multi-turn conversation",
        "Conversation round",
        "TTFT (ms)",
    )


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------
def scheduler_policy_bars(analysis: Analysis) -> go.Figure | None:
    frame = analysis.scheduler_frame()
    if frame.empty:
        return None
    frame = frame.dropna(subset=["ttft_ms_mean"])
    if frame.empty:
        return None
    frame = frame.sort_values("ttft_ms_mean")

    fig = go.Figure()
    for idx, row in frame.reset_index(drop=True).iterrows():
        policy = row.get("serving_mode")
        color = policy_color(str(policy), idx)
        label_text = row.get("extra_policy_label")
        if not isinstance(label_text, str) or not label_text.strip():
            label_text = policy
        fig.add_trace(
            go.Bar(
                x=[str(label_text)],
                y=[row["ttft_ms_mean"]],
                name=str(label_text or policy),
                marker_color=color,
                text=[f"{row['ttft_ms_mean']:.0f} ms"],
                textposition="outside",
                hovertemplate="%{x}<br>mean TTFT=%{y:.0f} ms<extra></extra>",
            )
        )
    fig.update_layout(showlegend=False)
    return _base_layout(
        fig, "Mean time-to-first-token by scheduling policy", "Policy", "Mean TTFT (ms)"
    )


def scheduler_group_split(analysis: Analysis) -> go.Figure | None:
    frame = analysis.scheduler_frame()
    if frame.empty or "extra_short_ttft_ms" not in frame.columns:
        return None
    frame = frame.dropna(subset=["extra_short_ttft_ms", "extra_long_ttft_ms"])
    if frame.empty:
        return None

    labels = [
        (
            row.get("extra_policy_label")
            if isinstance(row.get("extra_policy_label"), str)
            else str(row.get("serving_mode") or "?")
        )
        for _, row in frame.iterrows()
    ]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels,
            y=frame["extra_short_ttft_ms"],
            name="short requests",
            marker_color=PALETTE["emerald"],
            hovertemplate="%{x}<br>short TTFT=%{y:.0f} ms<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=labels,
            y=frame["extra_long_ttft_ms"],
            name="long requests",
            marker_color=PALETTE["rose"],
            hovertemplate="%{x}<br>long TTFT=%{y:.0f} ms<extra></extra>",
        )
    )
    fig.update_layout(barmode="group")
    return _base_layout(
        fig,
        "Fairness trade-off: short vs. long jobs",
        "Policy",
        "Mean TTFT (ms)",
    )


# ---------------------------------------------------------------------------
# Timeline & efficiency
# ---------------------------------------------------------------------------
def request_timeline(analysis: Analysis, max_requests: int = 60) -> go.Figure | None:
    """Gantt view: arrival → start (queueing) → first token → completion."""
    raw = analysis.raw
    if raw.empty or "start_ms" not in raw.columns:
        return None
    conc = raw[raw.get("scenario") == "concurrency"].dropna(subset=["start_ms", "e2e_ms"])
    if conc.empty:
        return None
    top = conc["concurrency"].max()
    conc = conc[conc["concurrency"] == top]
    backend = conc["series"].dropna().unique()
    if len(backend) == 0:
        return None
    backend = backend[0]
    conc = conc[conc["series"] == backend].sort_values("start_ms").head(max_requests)
    if conc.empty:
        return None

    fig = go.Figure()
    # Queueing segment: arrival → start.
    fig.add_trace(
        go.Bar(
            x=conc["queue_ms"].fillna(0.0),
            y=[f"#{int(i)}" for i in range(len(conc))],
            base=conc["arrival_ms"],
            orientation="h",
            name="queueing",
            marker_color=PALETTE["muted"],
            hovertemplate="queued %{x:.0f} ms<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=conc["e2e_ms"],
            y=[f"#{int(i)}" for i in range(len(conc))],
            base=conc["start_ms"],
            orientation="h",
            name="generation",
            marker_color=backend_color(str(backend)),
            hovertemplate="generating %{x:.0f} ms<extra></extra>",
        )
    )
    fig.update_layout(barmode="overlay", bargap=0.2, hovermode="closest")
    fig.update_yaxes(autorange="reversed", title="Request (sorted by start)")
    return _base_layout(
        fig,
        f"Request timeline — {backend}, concurrency={int(top)}",
        "Time since load start (ms)",
        "Request",
        height=max(320, min(720, 18 * len(conc))),
    )


def efficiency_pareto(analysis: Analysis) -> go.Figure | None:
    """Throughput vs. tail latency — the operating point that actually matters."""
    frame = analysis.concurrency_frame()
    if frame.empty or "throughput_tok_s" not in frame.columns or "e2e_ms_p99" not in frame.columns:
        return None
    frame = frame.dropna(subset=["throughput_tok_s", "e2e_ms_p99"])
    if frame.empty:
        return None

    fig = go.Figure()
    for idx, backend in enumerate(frame["series"].dropna().unique()):
        subset = frame[frame["series"] == backend].sort_values("concurrency")
        color = backend_color(backend, idx)
        fig.add_trace(
            go.Scatter(
                x=subset["e2e_ms_p99"],
                y=subset["throughput_tok_s"],
                mode="lines+markers+text",
                name=backend,
                text=[f"c={int(c)}" for c in subset["concurrency"]],
                textposition="top center",
                textfont={"size": 10, "color": PALETTE["slate"]},
                line={"color": color, "width": 2.4},
                marker={"size": 11, "color": color, "line": {"width": 2, "color": "white"}},
                hovertemplate=(
                    "p99 E2E=%{x:.0f} ms<br>throughput=%{y:.1f} tok/s<extra>%{fullData.name}</extra>"
                ),
            )
        )
    return _base_layout(
        fig,
        "Operating points: throughput vs. tail latency",
        "End-to-end p99 latency (ms)",
        "Throughput (tok/s)",
    )
