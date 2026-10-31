"""CLI surface and dashboard smoke tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llmbench.cli import build_parser, main


def test_parser_exposes_all_commands():
    parser = build_parser()
    actions = [a for a in parser._actions if a.dest == "command"]
    assert actions
    commands = set(actions[0].choices)
    assert {
        "run",
        "report",
        "dashboard",
        "sim",
        "runs",
        "show",
        "export",
        "datasets",
        "info",
        "fetch-model",
        "clean",
    } <= commands


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert "llmbench" in capsys.readouterr().out


def test_info_runs(capsys):
    assert main(["info"]) == 0
    out = capsys.readouterr().out
    assert "Runtime" in out or "python" in out
    assert "mock" in out


def test_datasets_command(tmp_path):
    out = tmp_path / "synthetic.jsonl"
    assert main(["datasets", "--out", str(out), "--num", "8"]) == 0
    assert out.exists()
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 8
    assert {"prompt", "target_length", "bucket"} <= set(rows[0])


def test_runs_command_empty(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("LLMBENCH_RESULTS_DIR", str(tmp_path / "empty"))
    assert main(["runs"]) == 0
    assert "No runs" in capsys.readouterr().out


def test_runs_and_show(synthetic_store, capsys):
    assert main(["runs"]) == 0
    out = capsys.readouterr().out
    assert "aaa-vllm" in out

    assert main(["show", "aaa-vllm", "--markdown"]) == 0
    out = capsys.readouterr().out
    assert "peak_throughput_tok_s" in out


def test_export_command(synthetic_store, tmp_path, capsys):
    out = tmp_path / "data.csv"
    assert main(["export", "--out", str(out)]) == 0
    assert out.exists()
    assert "scenario" in out.read_text().splitlines()[0]


def test_report_command(synthetic_store, tmp_path):
    assert main(["report", "--all", "--out-dir", str(tmp_path / "out"), "--no-html"]) == 0
    assert (tmp_path / "out" / "benchmark_report.generated.md").exists()


def test_sim_command(tmp_path, capsys):
    out = tmp_path / "sim.json"
    assert main(["sim", "--requests", "20", "--policies", "fcfs", "sjf", "--out", str(out)]) == 0
    payload = json.loads(out.read_text())
    assert len(payload) == 2
    assert all("ttft_p95_ms" in row for row in payload)


def test_clean_dry_run(synthetic_store, capsys):
    assert main(["clean", "--all", "--dry-run"]) == 0
    assert "would delete" in capsys.readouterr().out
    from llmbench.core.results import ResultStore

    assert len(ResultStore(synthetic_store).runs()) == 2


def test_invalid_set_override(synthetic_store, capsys):
    assert main(["run", "--set", "backend.name"]) == 2
    assert "KEY=VALUE" in capsys.readouterr().err


def test_bad_config_path(capsys):
    assert main(["run", "-c", "/no/such/config.yaml"]) == 2
    assert "Configuration error" in capsys.readouterr().err


def test_dashboard_app_loads(synthetic_store):
    """Execute the Streamlit script in-process and assert it renders."""
    from streamlit.testing.v1 import AppTest

    app = Path(__file__).resolve().parents[1] / "src" / "llmbench" / "dashboard" / "app.py"
    at = AppTest.from_file(str(app), default_timeout=90)
    at.run()
    assert not at.exception
    assert len(at.tabs) == 7
