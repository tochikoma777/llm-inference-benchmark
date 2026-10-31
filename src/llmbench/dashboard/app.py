"""Interactive Streamlit dashboard.

    llmbench dashboard                 # or: streamlit run src/llmbench/dashboard/app.py

The app reads the same on-disk artefacts as the CLI, so it never re-runs a
benchmark: pick one or more runs in the sidebar and explore.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow running the file directly from a source checkout.
try:  # pragma: no cover - import guard
    import llmbench  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import pandas as pd
import streamlit as st

from llmbench.core.paths import results_dir
from llmbench.core.results import ResultStore
from llmbench.reporting.analysis import Analysis
from llmbench.viz import plotly_charts as charts
from llmbench.viz.theme import prime_series_colors

# ---------------------------------------------------------------------------
# Presentation
# ---------------------------------------------------------------------------
_CSS = """
<style>
  .block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1400px;}
  #MainMenu, footer {visibility: hidden;}
  h1, h2, h3 {letter-spacing: -0.02em;}
  .hero {
      background: linear-gradient(120deg, #EFF4FF 0%, #F8FAFC 60%);
      border: 1px solid #E2E8F0; border-radius: 18px;
      padding: 26px 30px; margin-bottom: 18px;
  }
  .hero h1 {margin: 0 0 6px; font-size: 30px; color: #0F172A;}
  .hero p {margin: 0; color: #64748B; font-size: 15px;}
  .pill {
      display: inline-block; padding: 5px 12px; border-radius: 999px;
      background: #FFFFFF; border: 1px solid #E2E8F0; font-size: 12.5px;
      color: #334155; margin: 8px 8px 0 0;
  }
  .kpi {
      background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 16px;
      padding: 16px 18px; height: 100%;
      box-shadow: 0 1px 2px rgba(15,23,42,0.04);
  }
  .kpi .label {font-size: 12.5px; color: #64748B; font-weight: 600; letter-spacing: .01em;}
  .kpi .value {font-size: 27px; font-weight: 700; color: #0F172A; letter-spacing: -0.02em;
      font-variant-numeric: tabular-nums; line-height: 1.25; margin-top: 2px;}
  .kpi .hint {font-size: 12px; color: #94A3B8;}
  .kpi.blue .value {color: #2563EB;}
  .kpi.amber .value {color: #F59E0B;}
  .kpi.violet .value {color: #8B5CF6;}
  .kpi.emerald .value {color: #10B981;}
  .kpi.rose .value {color: #EF4444;}
  .warn {
      background: #FFF7ED; border: 1px solid #FED7AA; color: #9A3412;
      padding: 12px 16px; border-radius: 12px; font-size: 13.5px; margin-bottom: 16px;
  }
  .empty {
      background: #FFFFFF; border: 1px dashed #CBD5E1; border-radius: 16px;
      padding: 28px; text-align: center; color: #64748B;
  }
  div[data-testid="stMetricValue"] {font-size: 1.6rem;}
</style>
"""

_T: dict[str, dict[str, str]] = {
    "zh": {
        "title": "LLM 推理性能仪表盘",
        "subtitle": "跨后端、跨并发的吞吐与延迟对比 · 数据直接读取自 data/results/",
        "sidebar_data": "数据源",
        "sidebar_runs": "实验记录",
        "sidebar_refresh": "刷新",
        "no_runs": "还没有任何实验记录。先在终端执行：",
        "mock_warn": "当前选择包含模拟数据（mock 后端），数值由解析模型推导，仅用于演示。",
        "tab_overview": "总览",
        "tab_throughput": "吞吐",
        "tab_latency": "延迟",
        "tab_memory": "显存",
        "tab_prefix": "前缀缓存",
        "tab_scheduler": "调度",
        "tab_raw": "原始数据",
        "comparison": "后端对比（最高并发）",
        "records": "测量记录",
        "runs": "实验次数",
        "model": "模型",
        "gpu": "GPU",
        "download": "下载 CSV",
        "no_scenario": "本次运行没有该场景的数据。",
        "peak": "峰值吞吐",
        "speedup": "吞吐加速比",
        "ttft": "最优 TTFT p95",
        "memory": "KV Cache 节省",
        "prefix": "前缀缓存加速",
        "policy": "最优调度策略",
    },
    "en": {
        "title": "LLM Inference Dashboard",
        "subtitle": "Throughput and latency across backends and concurrency levels, read from data/results/",
        "sidebar_data": "Data source",
        "sidebar_runs": "Runs",
        "sidebar_refresh": "Refresh",
        "no_runs": "No runs yet. Start one from a terminal:",
        "mock_warn": "The current selection includes simulated data (mock backend). Numbers come from an analytic model.",
        "tab_overview": "Overview",
        "tab_throughput": "Throughput",
        "tab_latency": "Latency",
        "tab_memory": "Memory",
        "tab_prefix": "Prefix cache",
        "tab_scheduler": "Scheduler",
        "tab_raw": "Raw data",
        "comparison": "Backend comparison (highest concurrency)",
        "records": "measurements",
        "runs": "runs",
        "model": "Model",
        "gpu": "GPU",
        "download": "Download CSV",
        "no_scenario": "This run contains no data for that scenario.",
        "peak": "Peak throughput",
        "speedup": "Throughput speed-up",
        "ttft": "Best TTFT p95",
        "memory": "KV-cache saved",
        "prefix": "Prefix-cache speed-up",
        "policy": "Best policy",
    },
}


def _kpi_html(label: str, value: str, hint: str = "", accent: str = "blue") -> str:
    return (
        f'<div class="kpi {accent}"><div class="label">{label}</div>'
        f'<div class="value">{value}</div><div class="hint">{hint}</div></div>'
    )


def _fmt(value, digits: int = 1, suffix: str = "") -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    try:
        return f"{float(value):,.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return str(value)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _run_index(root: str, _stamp: float) -> list[dict]:
    store = ResultStore(root)
    out = []
    for info in store.runs():
        meta = info.metadata
        out.append(
            {
                "run_id": info.run_id,
                "label": f"{info.run_id[:15]} · {meta.backend} · {meta.model_id or 'n/a'}",
                "backend": meta.backend,
                "serving_mode": meta.serving_mode,
                "model": meta.model_id,
                "created_at": meta.created_at,
                "records": info.num_records,
                "status": meta.status,
                "is_mock": bool((meta.system or {}).get("mock")),
                "gpu": (meta.system or {}).get("gpu_name"),
                "tags": ",".join(meta.tags or []),
            }
        )
    return out


@st.cache_data(show_spinner=False)
def _load_analysis(root: str, run_ids: tuple, _stamp: float) -> Analysis:
    return Analysis.load(ResultStore(root), run_ids=list(run_ids))


def _dir_stamp(root: str) -> float:
    path = Path(root)
    if not path.exists():
        return 0.0
    stamps = [p.stat().st_mtime for p in path.glob("*/metadata.json")]
    return max(stamps) if stamps else 0.0


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
def _section_overview(analysis: Analysis, t: dict[str, str]) -> None:
    kpis = analysis.kpis()
    cols = st.columns(3)
    cards = []

    if kpis.get("peak_throughput_tok_s"):
        cards.append(
            _kpi_html(
                t["peak"],
                f"{kpis['peak_throughput_tok_s']:,.0f} tok/s",
                f"{kpis.get('peak_throughput_backend')} @ c={kpis.get('peak_concurrency')}",
                "blue",
            )
        )
    if kpis.get("throughput_speedup"):
        cards.append(
            _kpi_html(
                t["speedup"],
                f"{kpis['throughput_speedup']:.2f}×",
                f"{kpis.get('speedup_fastest','')} vs {kpis.get('speedup_baseline','')}",
                "emerald",
            )
        )
    if kpis.get("best_ttft_p95_ms") is not None:
        cards.append(_kpi_html(t["ttft"], _fmt(kpis["best_ttft_p95_ms"], 0, " ms"), "", "violet"))
    if kpis.get("memory_saving_pct") is not None:
        cards.append(
            _kpi_html(
                t["memory"],
                f"{kpis['memory_saving_pct']:.0f}%",
                f"{kpis.get('memory_best_backend','')}",
                "amber",
            )
        )
    if kpis.get("prefix_speedup"):
        cards.append(_kpi_html(t["prefix"], f"{kpis['prefix_speedup']:.2f}×", "", "emerald"))
    if kpis.get("best_policy"):
        cards.append(
            _kpi_html(
                t["policy"],
                str(kpis["best_policy"]).upper(),
                _fmt(kpis.get("best_policy_ttft_ms"), 0, " ms"),
                "rose",
            )
        )

    for index, card in enumerate(cards):
        with cols[index % 3]:
            st.markdown(card, unsafe_allow_html=True)
            st.write("")

    st.subheader(t["comparison"])
    comparison = analysis.comparison()
    if comparison.empty:
        st.info(t["no_scenario"])
    else:
        st.dataframe(comparison, width="stretch", hide_index=True)

    figure = charts.efficiency_pareto(analysis)
    if figure is not None:
        st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})


def _render_chart(analysis: Analysis, builder, t: dict[str, str], key: str) -> None:
    try:
        figure = builder(analysis)
    except Exception as exc:  # noqa: BLE001 - never break the whole page
        st.warning(f"Chart failed: {exc}")
        return
    if figure is None:
        st.info(t["no_scenario"])
        return
    st.plotly_chart(figure, width="stretch", key=key, config={"displayModeBar": False})


def _section_throughput(analysis: Analysis, t: dict[str, str]) -> None:
    left, right = st.columns(2)
    with left:
        _render_chart(analysis, charts.throughput_vs_concurrency, t, "tp")
    with right:
        _render_chart(analysis, charts.throughput_speedup, t, "sp")
    _render_chart(analysis, charts.efficiency_pareto, t, "pareto")
    frame = analysis.concurrency_frame()
    if not frame.empty:
        st.dataframe(frame, width="stretch", hide_index=True)


def _section_latency(analysis: Analysis, t: dict[str, str]) -> None:
    left, right = st.columns(2)
    with left:
        _render_chart(analysis, charts.latency_vs_concurrency, t, "lat")
    with right:
        _render_chart(analysis, charts.latency_cdf, t, "cdf")
    _render_chart(analysis, charts.request_timeline, t, "timeline")


def _section_memory(analysis: Analysis, t: dict[str, str]) -> None:
    left, right = st.columns(2)
    with left:
        _render_chart(analysis, charts.memory_vs_sequence_length, t, "mem")
    with right:
        _render_chart(analysis, charts.gpu_utilization, t, "util")
    left, right = st.columns(2)
    with left:
        _render_chart(analysis, charts.ttft_vs_prompt_length, t, "ttft")
    with right:
        _render_chart(analysis, charts.tpot_vs_prompt_length, t, "tpot")


def _section_prefix(analysis: Analysis, t: dict[str, str]) -> None:
    _render_chart(analysis, charts.prefix_cache_rounds, t, "prefix")
    frame = analysis.prefix_frame()
    if not frame.empty:
        st.dataframe(frame, width="stretch", hide_index=True)


def _section_scheduler(analysis: Analysis, t: dict[str, str]) -> None:
    left, right = st.columns(2)
    with left:
        _render_chart(analysis, charts.scheduler_policy_bars, t, "pol")
    with right:
        _render_chart(analysis, charts.scheduler_group_split, t, "split")
    frame = analysis.scheduler_frame()
    if not frame.empty:
        st.dataframe(frame, width="stretch", hide_index=True)


def _section_raw(analysis: Analysis, t: dict[str, str]) -> None:
    frame = analysis.raw
    st.caption(f"{len(frame):,} {t['records']}")
    scenarios = sorted(frame.get("scenario", pd.Series(dtype=str)).dropna().unique())
    chosen = st.multiselect("scenario", scenarios, default=scenarios)
    view = frame[frame["scenario"].isin(chosen)] if chosen else frame
    st.dataframe(view, width="stretch", hide_index=True, height=520)
    st.download_button(
        t["download"],
        data=view.to_csv(index=False).encode("utf-8"),
        file_name="llmbench_records.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
def run() -> None:
    st.set_page_config(
        page_title="LLM Inference Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_CSS, unsafe_allow_html=True)

    default_root = os.environ.get("LLMBENCH_RESULTS_DIR") or str(results_dir())

    with st.sidebar:
        st.markdown("### LLMBench")
        lang = st.radio("Language / 语言", ["中文", "English"], horizontal=True)
        t = _T["zh" if lang == "中文" else "en"]
        st.divider()
        st.caption(t["sidebar_data"])
        root = st.text_input("results dir", value=default_root, label_visibility="collapsed")
        if st.button(t["sidebar_refresh"], width="stretch"):
            _run_index.clear()
            _load_analysis.clear()

        stamp = _dir_stamp(root)
        index = _run_index(root, stamp)

        if not index:
            st.info(t["no_runs"])
            st.code("llmbench run -c config/smoke.yaml", language="bash")
            st.stop()

        labels = {item["run_id"]: item["label"] for item in index}
        chosen = st.multiselect(
            t["sidebar_runs"],
            options=list(labels),
            default=[index[0]["run_id"]],
            format_func=lambda key: labels[key],
        )
        st.divider()
        st.caption(f"{len(index)} run(s) · newest {index[0]['created_at'][:19].replace('T',' ')}")

    if not chosen:
        st.warning("Select at least one run in the sidebar.")
        st.stop()

    analysis = _load_analysis(root, tuple(chosen), stamp)
    if analysis.empty:
        st.warning("Selected runs contain no records.")
        st.stop()

    prime_series_colors(analysis.series())
    kpis = analysis.kpis()
    pills = []
    if kpis.get("model"):
        pills.append(f'<span class="pill">{t["model"]}: <b>{kpis["model"]}</b></span>')
    if kpis.get("gpu"):
        pills.append(f'<span class="pill">{t["gpu"]}: <b>{kpis["gpu"]}</b></span>')
    pills.append(f'<span class="pill">{t["runs"]}: <b>{kpis["runs"]}</b></span>')
    pills.append(f'<span class="pill">{t["records"]}: <b>{kpis["records"]:,}</b></span>')

    st.markdown(
        f'<div class="hero"><h1>{t["title"]}</h1><p>{t["subtitle"]}</p>' f'{"".join(pills)}</div>',
        unsafe_allow_html=True,
    )
    if analysis.is_mock:
        st.markdown(f'<div class="warn">{t["mock_warn"]}</div>', unsafe_allow_html=True)

    tabs = st.tabs(
        [
            t["tab_overview"],
            t["tab_throughput"],
            t["tab_latency"],
            t["tab_memory"],
            t["tab_prefix"],
            t["tab_scheduler"],
            t["tab_raw"],
        ]
    )
    with tabs[0]:
        _section_overview(analysis, t)
    with tabs[1]:
        _section_throughput(analysis, t)
    with tabs[2]:
        _section_latency(analysis, t)
    with tabs[3]:
        _section_memory(analysis, t)
    with tabs[4]:
        _section_prefix(analysis, t)
    with tabs[5]:
        _section_scheduler(analysis, t)
    with tabs[6]:
        _section_raw(analysis, t)


run()
