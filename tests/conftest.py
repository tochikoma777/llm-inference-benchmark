"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from llmbench.core.config import load_config
from llmbench.core.results import RunMetadata, RunWriter


@pytest.fixture()
def mock_config(tmp_path, monkeypatch):
    """A fast, deterministic mock configuration writing into ``tmp_path``."""
    monkeypatch.setenv("LLMBENCH_RESULTS_DIR", str(tmp_path / "results"))
    config = load_config("config/smoke.yaml")
    config.mock.time_scale = 0.0  # no wall-clock sleeping at all
    config.mock.noise = 0.0
    config.scenarios.concurrency_levels = [1, 2, 4]
    config.scenarios.prompt_lengths = [128, 512]
    config.scenarios.output_lengths = [1, 32]
    config.sim.num_requests = 20
    config.sim.policies = ["fcfs", "sjf"]
    config.sim.max_num_seqs = 4
    return config


@pytest.fixture()
def results_dir(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "results"
    path.mkdir()
    monkeypatch.setenv("LLMBENCH_RESULTS_DIR", str(path))
    return path


def write_synthetic_run(
    root: Path,
    run_id: str = "test-run",
    backend: str = "vllm",
    serving_mode: str = "continuous_batch",
    scenarios=("concurrency",),
    mock: bool = False,
) -> Path:
    """Write a small, hand-crafted run for analysis tests."""
    meta = RunMetadata(
        run_id=run_id,
        name=run_id,
        backend=backend,
        serving_mode=serving_mode,
        model_id="Qwen2.5-7B-Instruct",
        system={"gpu_name": "RTX 3090", "mock": mock, "platform": "test"},
    )
    writer = RunWriter(meta, directory=root / run_id)

    if "concurrency" in scenarios:
        for concurrency in (1, 4):
            for index in range(4):
                ttft = 100.0 / concurrency
                e2e = ttft + 900.0
                writer.add(
                    {
                        "scenario": "concurrency",
                        "level": "request",
                        "backend": backend,
                        "serving_mode": serving_mode,
                        "bucket": "load",
                        "request_index": index,
                        "concurrency": concurrency,
                        "prompt_len": 256,
                        "output_len": 64,
                        "input_tokens": 256,
                        "output_tokens": 64,
                        "arrival_ms": 0.0,
                        "start_ms": 0.0,
                        "ttft_ms": ttft,
                        "tpot_ms": e2e / 64,
                        "e2e_ms": e2e,
                        "finish_ms": e2e,
                        "throughput_tok_s": 64 / (e2e / 1000),
                        "peak_memory_mb": 15000.0,
                        "avg_gpu_util_pct": 80.0,
                        "success": True,
                    }
                )

    if "memory" in scenarios:
        for prompt_len in (256, 2048):
            writer.add(
                {
                    "scenario": "memory",
                    "level": "aggregate",
                    "backend": backend,
                    "serving_mode": serving_mode,
                    "bucket": f"mem_p{prompt_len}",
                    "group": f"mem_p{prompt_len}",
                    "concurrency": 1,
                    "prompt_len": prompt_len,
                    "output_len": 128,
                    "peak_memory_mb": 14000.0 + prompt_len * 0.2,
                    "kv_cache_delta_mb": 100.0 + prompt_len * 0.2,
                    "avg_gpu_util_pct": 70.0,
                    "success": True,
                }
            )

    if "prefix_cache" in scenarios:
        for cache in ("cache_off", "cache_on"):
            for round_index in (1, 2, 3):
                writer.add(
                    {
                        "scenario": "prefix_cache",
                        "level": "request",
                        "backend": backend,
                        "serving_mode": serving_mode,
                        "bucket": f"{cache}_r{round_index}",
                        "group": f"{cache}_r{round_index}",
                        "concurrency": 1,
                        "prompt_len": 2000 + round_index * 50,
                        "output_len": 32,
                        "output_tokens": 32,
                        "ttft_ms": 400.0 if cache == "cache_off" else 400.0 / round_index,
                        "e2e_ms": 900.0,
                        "finish_ms": 900.0,
                        "success": True,
                    }
                )

    if "scheduler" in scenarios:
        for policy, label, short_ttft, long_ttft in (
            ("fcfs", "FCFS", 6000.0, 7000.0),
            ("sjf", "SJF", 1500.0, 8000.0),
        ):
            writer.add(
                {
                    "scenario": "scheduler",
                    "level": "aggregate",
                    "backend": "simulation",
                    "serving_mode": policy,
                    "bucket": policy,
                    "group": policy,
                    "concurrency": 8,
                    "num_requests": 100,
                    "ttft_ms": (short_ttft + long_ttft) / 2,
                    "ttft_ms_mean": (short_ttft + long_ttft) / 2,
                    "ttft_ms_p95": long_ttft,
                    "e2e_ms": 20000.0,
                    "e2e_ms_mean": 20000.0,
                    "e2e_ms_p99": 60000.0,
                    "throughput_tok_s": 150.0,
                    "extra_policy_label": label,
                    "extra_short_ttft_ms": short_ttft,
                    "extra_long_ttft_ms": long_ttft,
                    "extra_preemptions": 0,
                    "extra_fairness_jain": 0.5,
                    "success": True,
                }
            )

    writer.write()
    return root / run_id


@pytest.fixture()
def make_run(results_dir):
    """Factory fixture: write an extra synthetic run into the temp store.

    Exposed as a fixture rather than a helper function so tests never import
    from ``tests.conftest`` — that only works when the repository root happens
    to be on ``sys.path`` (i.e. with ``python -m pytest``, not with the
    ``pytest`` console script used in CI).
    """

    def _make(**kwargs) -> Path:
        kwargs.setdefault("run_id", "extra-run")
        return write_synthetic_run(results_dir, **kwargs)

    return _make


@pytest.fixture()
def synthetic_store(results_dir) -> Path:
    write_synthetic_run(
        results_dir,
        run_id="aaa-vllm",
        backend="vllm",
        serving_mode="continuous_batch",
        scenarios=("concurrency", "memory", "prefix_cache", "scheduler"),
    )
    write_synthetic_run(
        results_dir,
        run_id="bbb-naive",
        backend="naive",
        serving_mode="static_batch",
        scenarios=("concurrency", "memory"),
    )
    return results_dir
