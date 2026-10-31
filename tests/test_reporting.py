"""Analysis, reporting and end-to-end pipeline tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from llmbench.core.results import ResultStore
from llmbench.reporting.analysis import Analysis
from llmbench.reporting.html_report import build_html_report
from llmbench.reporting.markdown import build_markdown_report, dataframe_to_markdown


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def test_analysis_loads_two_runs(synthetic_store):
    analysis = Analysis.load(ResultStore(synthetic_store))
    assert not analysis.empty
    assert analysis.backends() == ["naive", "simulation", "vllm"]
    assert len(analysis.runs) == 2


def test_analysis_series_distinguishes_serving_modes(synthetic_store):
    analysis = Analysis.load(ResultStore(synthetic_store))
    series = analysis.series()
    assert "vllm" in series
    assert "naive · static_batch" in series


def test_kpis_are_computed(synthetic_store):
    analysis = Analysis.load(ResultStore(synthetic_store))
    kpis = analysis.kpis()
    assert kpis["peak_throughput_tok_s"] > 0
    assert kpis["throughput_speedup"] is not None
    assert kpis["memory_saving_pct"] is not None
    assert 0 <= kpis["memory_saving_pct"] < 100
    assert kpis["prefix_speedup"] is not None
    assert kpis["best_policy"] == "sjf"


def test_comparison_picks_highest_concurrency(synthetic_store):
    analysis = Analysis.load(ResultStore(synthetic_store))
    table = analysis.comparison()
    assert not table.empty
    assert set(table["Concurrency"]) == {4}
    assert "Speed-up" in table.columns


def test_concurrency_frame_has_series(synthetic_store):
    analysis = Analysis.load(ResultStore(synthetic_store))
    frame = analysis.concurrency_frame()
    assert {"series", "concurrency", "throughput_tok_s"} <= set(frame.columns)
    assert len(frame) == 4  # 2 series × 2 levels


def test_scheduler_frame_prefers_simulator_rows(synthetic_store):
    analysis = Analysis.load(ResultStore(synthetic_store))
    frame = analysis.scheduler_frame()
    assert len(frame) == 2
    assert frame["extra_short_ttft_ms"].notna().all()


def test_memory_and_prefix_frames(synthetic_store):
    analysis = Analysis.load(ResultStore(synthetic_store))
    assert len(analysis.memory_frame()) == 4
    assert len(analysis.prefix_frame()) == 6


def test_repeated_runs_are_averaged_not_summed(synthetic_store, make_run):
    """Two identical runs must not double the reported throughput."""
    store = ResultStore(synthetic_store)
    before = Analysis.load(store).concurrency_frame()
    baseline = before[before["concurrency"] == 4]["throughput_tok_s"].iloc[0]

    make_run(
        run_id="ccc-vllm-repeat",
        backend="vllm",
        serving_mode="continuous_batch",
        scenarios=("concurrency",),
    )
    after = Analysis.load(store).concurrency_frame()
    repeated = after[(after["concurrency"] == 4) & (after["series"] == "vllm")]
    assert repeated["throughput_tok_s"].iloc[0] == pytest.approx(baseline, rel=1e-6)


def test_analysis_empty_store(tmp_path):
    analysis = Analysis.load(ResultStore(tmp_path))
    assert analysis.empty
    assert analysis.kpis()["records"] == 0
    assert analysis.comparison().empty


def test_scaling_efficiency(synthetic_store):
    analysis = Analysis.load(ResultStore(synthetic_store))
    frame = analysis.scaling_efficiency()
    assert "efficiency_tok_s_per_stream" in frame.columns
    assert (frame["efficiency_tok_s_per_stream"] >= 0).all()


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
def test_markdown_table_rendering():
    import pandas as pd

    text = dataframe_to_markdown(pd.DataFrame({"a": [1, 2], "b": [0.5, None]}))
    assert text.startswith("| a | b |")
    assert "| 1 | 0.50 |" in text
    assert "—" in text


def test_markdown_report_written(synthetic_store, tmp_path):
    analysis = Analysis.load(ResultStore(synthetic_store))
    path = build_markdown_report(analysis, tmp_path / "report.md")
    content = path.read_text(encoding="utf-8")
    assert "# LLM Inference Benchmark" in content
    assert "关键指标" in content
    assert "调度策略对比" in content


def test_html_report_is_self_contained(synthetic_store, tmp_path):
    analysis = Analysis.load(ResultStore(synthetic_store))
    path = build_html_report(analysis, tmp_path / "report.html", lang="en")
    html = path.read_text(encoding="utf-8")

    assert path.stat().st_size > 100_000
    assert "plotly-graph-div" in html
    assert 'src="https://cdn.plot.ly' not in html  # plotly.js must be inlined
    assert html.count("<script") >= 2
    assert ">None<" not in html
    # Sidecar JSON with the KPIs is written next to the report.
    assert (tmp_path / "report.json").exists()


def test_html_report_survives_missing_scenarios(synthetic_store, tmp_path):
    store = ResultStore(synthetic_store)
    analysis = Analysis.load(store, run_ids=["bbb-naive"])
    path = build_html_report(analysis, tmp_path / "partial.html", lang="zh")
    assert path.exists()
    html = path.read_text(encoding="utf-8")
    assert "plotly-graph-div" in html
    assert ">None<" not in html


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def test_figures_are_rendered(synthetic_store, tmp_path):
    from llmbench.viz.figures import make_all_figures

    analysis = Analysis.load(ResultStore(synthetic_store))
    written = make_all_figures(analysis, tmp_path / "figures", lang="en")
    assert len(written) >= 5
    for path in written:
        assert Path(path).exists()
        assert Path(path).stat().st_size > 5_000


def test_theme_language_falls_back_when_no_cjk_font():
    from llmbench.viz.theme import has_cjk_font, resolve_language

    resolved = resolve_language("zh")
    assert resolved in {"zh", "en"}
    if not has_cjk_font():
        assert resolved == "en"


def test_series_colours_are_stable():
    from llmbench.viz.theme import backend_color, prime_series_colors

    prime_series_colors(["vllm", "naive · static_batch", "mock · per_request"])
    first = backend_color("vllm")
    backend_color("naive · static_batch")
    assert backend_color("vllm") == first
    assert backend_color("vllm") != backend_color("naive · static_batch")


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_full_pipeline_with_mock(mock_config, tmp_path):
    from llmbench.reporting.analysis import Analysis
    from llmbench.runner.executive import run_suite

    path = run_suite(mock_config)
    assert (path / "results.jsonl").exists()

    analysis = Analysis.load(ResultStore(path.parent))
    assert not analysis.empty
    assert analysis.is_mock
    scenarios = set(analysis.raw["scenario"])
    assert {"single", "concurrency", "memory", "prefix_cache", "scheduler"} <= scenarios


@pytest.mark.slow
def test_selected_scenarios_only(mock_config):
    from llmbench.runner.executive import run_suite

    path = run_suite(mock_config, scenarios=["concurrency"])
    lines = (path / "results.jsonl").read_text().strip().splitlines()
    assert lines
    assert all('"scenario": "concurrency"' in line for line in lines)


def test_unknown_scenario_rejected(mock_config):
    from llmbench.runner.executive import run_suite

    with pytest.raises(ValueError):
        run_suite(mock_config, scenarios=["nope"])
