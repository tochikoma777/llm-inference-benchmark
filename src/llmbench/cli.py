"""``llmbench`` command line interface.

llmbench run -c config/vllm.yaml --model-path /models/Qwen2.5-7B-Instruct
llmbench report --latest --open
llmbench dashboard
llmbench sim --policies fcfs sjf
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import webbrowser
from pathlib import Path
from typing import Any

from . import __version__
from .core.config import ConfigError, load_config
from .core.logging import get_logger, setup_logging
from .core.paths import (
    data_dir,
    display_path,
    figures_dir,
    logs_dir,
    project_root,
    prompts_dir,
    reports_dir,
    results_dir,
)
from .core.results import ResultStore

__all__ = ["main", "build_parser"]

log = get_logger("cli")


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def _console():
    try:
        from rich.console import Console

        return Console()
    except Exception:  # pragma: no cover
        return None


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def _rule(title: str) -> None:
    console = _console()
    if console:
        console.rule(f"[bold]{title}")
    else:  # pragma: no cover
        print(f"\n=== {title} ===")


def _table(title: str, columns: list[str], rows: list[list]) -> None:
    console = _console()
    if console is not None:
        try:
            from rich.table import Table

            table = Table(title=title, header_style="bold", title_style="bold", show_lines=False)
            for column in columns:
                table.add_column(column, overflow="fold")
            for row in rows:
                table.add_row(*[str(cell) for cell in row])
            console.print(table)
            return
        except Exception:  # pragma: no cover
            pass
    print(f"\n{title}")
    print(" | ".join(columns))
    for row in rows:
        print(" | ".join(str(cell) for cell in row))


def _fmt_concurrency(value: Any) -> str:
    """Render 4.0 as ``4`` so CLI tables stay tidy."""
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return str(int(number)) if number.is_integer() else f"{number:g}"


def _fmt(value: Any, digits: int = 1, suffix: str = "") -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):,.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return str(value)


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
def cmd_run(args: argparse.Namespace) -> int:
    overrides: dict[str, Any] = {}
    if args.backend:
        overrides["backend.name"] = args.backend
    if args.serving_mode:
        overrides["backend.serving_mode"] = args.serving_mode
    if args.model_path:
        overrides["model.path"] = args.model_path
    if args.name:
        overrides["name"] = args.name
    if args.output_dir:
        overrides["output.dir"] = args.output_dir
    if args.time_scale is not None:
        overrides["mock.time_scale"] = str(args.time_scale)
    if args.gpu_profile:
        overrides["mock.gpu_profile"] = args.gpu_profile
    if args.concurrency:
        overrides["scenarios.concurrency_levels"] = args.concurrency
    if args.max_new_tokens:
        overrides["workload.max_new_tokens"] = args.max_new_tokens
    for item in args.set or []:
        if "=" not in item:
            raise ConfigError(f"--set expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        overrides[key.strip()] = value.strip()

    # Backend switch implies a sensible serving mode unless explicitly set.
    if args.backend and not args.serving_mode:
        overrides["backend.serving_mode"] = "auto"

    config = load_config(args.config, overrides)
    if args.scenarios:
        for name in ("single", "concurrency", "memory", "prefix_cache", "scheduler"):
            setattr(config.scenarios, name, name in args.scenarios)
        if not any(
            getattr(config.scenarios, n)
            for n in ("single", "concurrency", "memory", "prefix_cache", "scheduler")
        ):
            raise ConfigError(f"--scenarios must include one of {args.scenarios!r}")

    if args.output_dir:
        os.environ["LLMBENCH_RESULTS_DIR"] = str(Path(args.output_dir).expanduser())

    _rule(f"llmbench · {config.name}")
    console = _console()
    if console:
        console.print(
            f"backend=[bold]{config.backend.name}[/bold] "
            f"mode=[bold]{config.backend.effective_serving_mode}[/bold] "
            f"model=[bold]{config.model.path or '(mock)'}[/bold]"
        )

    from .runner.executive import run_suite

    path = run_suite(config, scenarios=args.scenarios or None)

    store = ResultStore()
    analysis = _load_analysis(store, [path.name])
    if analysis is not None and not analysis.empty:
        kpis = analysis.kpis()
        _table(
            "Key metrics",
            ["Metric", "Value", "Context"],
            [
                [
                    "Peak throughput",
                    f"{_fmt(kpis.get('peak_throughput_tok_s'), 0)} tok/s",
                    f"{kpis.get('peak_throughput_backend', '—')} @ "
                    f"c={_fmt_concurrency(kpis.get('peak_concurrency'))}",
                ],
                [
                    "Throughput speed-up",
                    f"{_fmt(kpis.get('throughput_speedup'), 2)}×",
                    (
                        f"{kpis.get('speedup_fastest', '—')} vs {kpis.get('speedup_baseline', '—')}"
                        if kpis.get("speedup_baseline")
                        else "single configuration"
                    ),
                ],
                ["Best TTFT p95", f"{_fmt(kpis.get('best_ttft_p95_ms'), 0)} ms", ""],
                [
                    "KV cache saved",
                    f"{_fmt(kpis.get('memory_saving_pct'), 0)}%",
                    str(kpis.get("memory_best_backend", "")),
                ],
                ["Prefix-cache speed-up", f"{_fmt(kpis.get('prefix_speedup'), 2)}×", ""],
                ["Best policy", str(kpis.get("best_policy") or "—"), ""],
            ],
        )

    print(f"\n✓ Run saved to {display_path(path)}")
    print(f"  Next:  llmbench report --run-id {path.name}")
    return 0


def _load_analysis(store: ResultStore, run_ids: list[str] | None = None):
    from .reporting.analysis import Analysis

    try:
        return Analysis.load(store, run_ids=run_ids)
    except Exception as exc:  # pragma: no cover
        log.warning("Could not analyse results: %s", exc)
        return None


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def cmd_report(args: argparse.Namespace) -> int:
    store = ResultStore()
    run_ids = list(args.run_id) if args.run_id else None
    if args.all:
        run_ids = None
    elif run_ids is None:
        latest = store.latest(1)
        if not latest:
            print("No runs found. Execute `llmbench run` first.", file=sys.stderr)
            return 1
        run_ids = [latest[0].run_id]

    from .reporting.analysis import Analysis

    analysis = Analysis.load(store, run_ids=run_ids)
    if analysis.empty:
        print("No records found for the selected run(s).", file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir).expanduser() if args.out_dir else reports_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    if not args.no_figures:
        from .viz.figures import make_all_figures

        for figure_path in make_all_figures(analysis, out_dir / "figures", lang=args.lang):
            written.append(Path(figure_path))

    if not args.no_html:
        from .reporting.html_report import build_html_report, report_filename

        html_path: Path = Path(args.html_out) if args.html_out else out_dir / report_filename()
        written.append(
            build_html_report(
                analysis, html_path, lang="zh" if args.lang in ("auto", "zh") else "en"
            )
        )

    if not args.no_markdown:
        from .reporting.markdown import build_markdown_report

        written.append(build_markdown_report(analysis, out_dir / "benchmark_report.generated.md"))

    _rule("Report generated")
    for artifact in written:
        print(f"  • {display_path(artifact)}")

    html_files = [p for p in written if p.suffix == ".html"]
    if html_files and args.open:
        webbrowser.open(html_files[0].resolve().as_uri())
    return 0


# ---------------------------------------------------------------------------
# dashboard
# ---------------------------------------------------------------------------
def cmd_dashboard(args: argparse.Namespace) -> int:
    app = Path(__file__).resolve().parent / "dashboard" / "app.py"
    if not app.exists():
        print(f"Dashboard app not found at {app}", file=sys.stderr)
        return 1
    env = dict(os.environ)
    if args.results_dir:
        env["LLMBENCH_RESULTS_DIR"] = str(Path(args.results_dir).expanduser())
    env.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app),
        "--server.port",
        str(args.port),
        "--server.headless",
        "false" if args.open else "true",
        "--browser.gatherUsageStats",
        "false",
    ]
    print("Launching dashboard:", " ".join(cmd))
    try:
        return subprocess.call(cmd, env=env)
    except FileNotFoundError:
        print(
            "Streamlit is not installed. Install the dashboard extras:\n"
            "  pip install -r requirements-viz.txt",
            file=sys.stderr,
        )
        return 1


# ---------------------------------------------------------------------------
# sim
# ---------------------------------------------------------------------------
def cmd_sim(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if args.requests:
        config.sim.num_requests = args.requests
    if args.policies:
        config.sim.policies = args.policies
    if args.seed is not None:
        config.sim.seed = args.seed
    if args.max_num_seqs:
        config.sim.max_num_seqs = args.max_num_seqs
    config.sim.validate()

    from .scheduler_sim.runner import compare_policies

    _rule("Scheduler simulation")
    results = compare_policies(config)
    rows = [
        [
            result.label,
            _fmt(result.metrics["ttft_mean_ms"], 1),
            _fmt(result.metrics["ttft_p95_ms"], 1),
            _fmt(result.metrics["e2e_p99_ms"], 1),
            _fmt(result.metrics["throughput_tok_s"], 1),
            _fmt(result.metrics["wait_max_ms"], 1),
            int(result.metrics["preemptions"]),
            _fmt(result.metrics["fairness_jain"], 3),
        ]
        for result in results
    ]
    _table(
        "Policy comparison",
        ["Policy", "TTFT mean", "TTFT p95", "E2E p99", "Tok/s", "Max wait", "Preempt", "Fairness"],
        rows,
    )

    if args.out:
        payload = [{"policy": r.policy, "label": r.label, **r.metrics} for r in results]
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nSaved to {display_path(out)}")
    return 0


# ---------------------------------------------------------------------------
# runs / show / export
# ---------------------------------------------------------------------------
def cmd_runs(args: argparse.Namespace) -> int:
    store = ResultStore(args.results_dir)
    infos = store.runs()
    if not infos:
        print(f"No runs in {display_path(store.root)}")
        return 0
    rows = []
    for info in infos:
        meta = info.metadata
        rows.append(
            [
                info.run_id,
                meta.backend or "—",
                meta.serving_mode or "—",
                meta.model_id or "—",
                meta.status,
                info.num_records,
                (meta.created_at or "")[:19].replace("T", " "),
            ]
        )
    _table(
        "Benchmark runs",
        ["Run ID", "Backend", "Mode", "Model", "Status", "Records", "Created"],
        rows,
    )
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    store = ResultStore(args.results_dir)
    run_id = args.run_id
    if not run_id:
        latest = store.latest(1)
        if not latest:
            print("No runs found.", file=sys.stderr)
            return 1
        run_id = latest[0].run_id

    from .reporting.analysis import Analysis
    from .reporting.markdown import dataframe_to_markdown

    analysis = Analysis.load(store, run_ids=[run_id])
    if analysis.empty:
        print(f"No records in run {run_id}", file=sys.stderr)
        return 1

    kpis = analysis.kpis()
    _rule(f"Run {run_id}")
    for key in (
        "model",
        "gpu",
        "backends",
        "scenarios",
        "peak_throughput_tok_s",
        "throughput_speedup",
        "best_ttft_p95_ms",
        "memory_saving_pct",
        "prefix_speedup",
        "best_policy",
        "is_mock",
    ):
        if key in kpis and kpis[key] is not None:
            print(f"  {key:24s} {kpis[key]}")

    if args.markdown:
        print()
        print(dataframe_to_markdown(analysis.comparison()))
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    store = ResultStore(args.results_dir)
    out = Path(args.out) if args.out else data_dir() / "export.csv"
    path = store.export_csv(out, run_ids=args.run_id or None)
    print(f"Exported to {display_path(path)}")
    return 0


# ---------------------------------------------------------------------------
# datasets / info / fetch-model / clean
# ---------------------------------------------------------------------------
def cmd_datasets(args: argparse.Namespace) -> int:
    from .runner.workload import write_synthetic_dataset

    out = Path(args.out) if args.out else prompts_dir() / "synthetic.jsonl"
    path = write_synthetic_dataset(out, num_prompts=args.num, seed=args.seed)
    print(f"Wrote dataset to {display_path(path)}")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    from .backends import available_backends
    from .metrics.system import collect_system_info

    info = collect_system_info()
    _rule("Environment")
    rows = [[k, v] for k, v in info.items() if not isinstance(v, dict)]
    for name, version in (info.get("packages") or {}).items():
        rows.append([f"package:{name}", version or "not installed"])
    _table("Runtime", ["Key", "Value"], rows)

    _rule("Paths")
    _table(
        "Locations",
        ["Name", "Path"],
        [
            ["project root", str(project_root())],
            ["data", display_path(data_dir())],
            ["prompts", display_path(prompts_dir())],
            ["results", display_path(results_dir())],
            ["reports", display_path(reports_dir())],
            ["figures", display_path(figures_dir())],
            ["logs", display_path(logs_dir())],
        ],
    )

    _rule("Backends")
    _table("Available", ["Name", "Description"], [[k, v] for k, v in available_backends().items()])
    return 0


def cmd_fetch_model(args: argparse.Namespace) -> int:
    try:
        from modelscope import snapshot_download
    except ImportError:
        print(
            "modelscope is required:\n  pip install -r requirements-gpu.txt",
            file=sys.stderr,
        )
        return 1

    cache = Path(args.cache_dir).expanduser() if args.cache_dir else Path("~/models").expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {args.model} → {cache}")
    path = snapshot_download(args.model, cache_dir=str(cache))
    print(f"\n✓ Model available at: {path}")
    print(f"  export LLMBENCH_MODEL_PATH={path}")
    return 0


def cmd_clean(args: argparse.Namespace) -> int:
    import time

    store = ResultStore(args.results_dir)
    infos = store.runs()
    if not infos:
        print("Nothing to clean.")
        return 0
    cutoff = time.time() - args.older_than * 86400 if args.older_than else None
    targets = [info for info in infos if args.all or cutoff is None or info.mtime < cutoff]
    if not targets:
        print("No runs matched the filter.")
        return 0
    for info in targets:
        if args.dry_run:
            print(f"[dry-run] would delete {info.run_id}")
        else:
            store.delete(info.run_id)
            print(f"deleted {info.run_id}")
    return 0


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llmbench",
        description="Reproducible benchmarking for LLM serving backends.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  llmbench run -c config/smoke.yaml\n"
            "  llmbench run -c config/vllm.yaml --model-path /models/Qwen2.5-7B-Instruct\n"
            "  llmbench report --latest --open\n"
            "  llmbench dashboard --port 8501\n"
            "  llmbench sim --policies fcfs sjf priority\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"llmbench {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="enable debug logging")
    parser.add_argument("-q", "--quiet", action="store_true", help="only show warnings and errors")

    subs = parser.add_subparsers(dest="command", required=True)

    p_run = subs.add_parser("run", help="run the benchmark suite")
    p_run.add_argument(
        "-c", "--config", default=None, help="YAML/JSON config (default: config/default.yaml)"
    )
    p_run.add_argument("--backend", choices=["mock", "naive", "vllm"])
    p_run.add_argument(
        "--serving-mode",
        choices=["auto", "continuous_batch", "static_batch", "per_request", "offline", "async"],
    )
    p_run.add_argument("--model-path")
    p_run.add_argument(
        "--scenarios",
        nargs="+",
        choices=["single", "concurrency", "memory", "prefix_cache", "scheduler"],
    )
    p_run.add_argument(
        "--concurrency", nargs="+", type=int, help="concurrency levels, e.g. 1 2 4 8"
    )
    p_run.add_argument("--max-new-tokens", type=int)
    p_run.add_argument("--name", help="human readable run name")
    p_run.add_argument("--tag", action="append", help="tag (repeatable)")
    p_run.add_argument(
        "--set", action="append", metavar="KEY=VALUE", help="override any config key"
    )
    p_run.add_argument("--output-dir", help="where to store the run")
    p_run.add_argument("--time-scale", type=float, help="mock backend wall-clock scaling")
    p_run.add_argument(
        "--gpu-profile",
        choices=["rtx3090", "a100", "h100", "cpu"],
        help="mock backend hardware profile",
    )
    p_run.set_defaults(func=cmd_run)

    p_report = subs.add_parser("report", help="build HTML/Markdown reports and figures")
    p_report.add_argument("--run-id", action="append", help="run id (repeatable)")
    p_report.add_argument("--latest", action="store_true", help="use the most recent run")
    p_report.add_argument("--all", action="store_true", help="combine every stored run")
    p_report.add_argument("--out-dir", help="output directory (default: reports/)")
    p_report.add_argument("--html-out", help="explicit HTML output path")
    p_report.add_argument("--lang", choices=["auto", "zh", "en"], default="auto")
    p_report.add_argument("--no-figures", action="store_true")
    p_report.add_argument("--no-html", action="store_true")
    p_report.add_argument("--no-markdown", action="store_true")
    p_report.add_argument("--open", action="store_true", help="open the HTML report in a browser")
    p_report.set_defaults(func=cmd_report)

    p_dash = subs.add_parser("dashboard", help="launch the interactive Streamlit dashboard")
    p_dash.add_argument("--port", type=int, default=8501)
    p_dash.add_argument("--results-dir")
    p_dash.add_argument("--open", action="store_true", help="open a browser window")
    p_dash.set_defaults(func=cmd_dashboard)

    p_sim = subs.add_parser("sim", help="run the scheduler-policy simulation")
    p_sim.add_argument("-c", "--config", default=None)
    p_sim.add_argument("--requests", type=int)
    p_sim.add_argument(
        "--policies", nargs="+", choices=["fcfs", "sjf", "priority", "sjf_aging", "round_robin"]
    )
    p_sim.add_argument("--max-num-seqs", type=int, dest="max_num_seqs")
    p_sim.add_argument("--seed", type=int)
    p_sim.add_argument("--out", help="write metrics as JSON")
    p_sim.set_defaults(func=cmd_sim)

    p_runs = subs.add_parser("runs", help="list stored benchmark runs")
    p_runs.add_argument("--results-dir")
    p_runs.set_defaults(func=cmd_runs)

    p_show = subs.add_parser("show", help="show headline metrics of a run")
    p_show.add_argument("run_id", nargs="?")
    p_show.add_argument("--results-dir")
    p_show.add_argument("--markdown", action="store_true", help="also print the comparison table")
    p_show.set_defaults(func=cmd_show)

    p_export = subs.add_parser("export", help="export measurements to CSV")
    p_export.add_argument("--run-id", action="append")
    p_export.add_argument("--out")
    p_export.add_argument("--results-dir")
    p_export.set_defaults(func=cmd_export)

    p_ds = subs.add_parser("datasets", help="generate a synthetic prompt dataset")
    p_ds.add_argument("--out")
    p_ds.add_argument("--num", type=int, default=120)
    p_ds.add_argument("--seed", type=int, default=42)
    p_ds.set_defaults(func=cmd_datasets)

    p_info = subs.add_parser("info", help="print environment, paths and backends")
    p_info.set_defaults(func=cmd_info)

    p_fetch = subs.add_parser("fetch-model", help="download a model from ModelScope")
    p_fetch.add_argument("--model", default="qwen/Qwen2.5-7B-Instruct")
    p_fetch.add_argument("--cache-dir")
    p_fetch.set_defaults(func=cmd_fetch_model)

    p_clean = subs.add_parser("clean", help="delete stored runs")
    p_clean.add_argument("--all", action="store_true")
    p_clean.add_argument("--older-than", type=float, metavar="DAYS")
    p_clean.add_argument("--results-dir")
    p_clean.add_argument("--dry-run", action="store_true")
    p_clean.set_defaults(func=cmd_clean)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    level = (
        "DEBUG"
        if getattr(args, "verbose", False)
        else "WARNING" if getattr(args, "quiet", False) else "INFO"
    )
    setup_logging(level)

    handler = getattr(args, "func", None)
    if handler is None:  # pragma: no cover - argparse enforces a subcommand
        parser.print_help()
        return 2
    try:
        return int(handler(args) or 0)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:  # pragma: no cover
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:  # pragma: no cover - top-level guard
        if getattr(args, "verbose", False):
            raise
        print(f"Error: {exc}", file=sys.stderr)
        print("Re-run with -v for a traceback.", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
