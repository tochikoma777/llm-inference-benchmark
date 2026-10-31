"""Mock backend + load generator behaviour."""

from __future__ import annotations

import pytest

from llmbench.backends import available_backends, create_backend
from llmbench.backends.base import GenerationOutput, GenerationRequest
from llmbench.backends.mock_backend import PROFILES, MockBackend
from llmbench.core.config import load_config
from llmbench.runner.loadgen import plan_arrivals, run_batch, run_load, summarize_load


def _config(mode="continuous_batch", time_scale: float = 0.0, **overrides):
    config = load_config(overrides=[f"backend.serving_mode={mode}"])
    config.backend.name = "mock"
    config.mock.time_scale = time_scale
    config.mock.noise = 0.0
    for key, value in overrides.items():
        setattr(config.mock, key, value)
    return config


def test_registry_lists_all_backends():
    assert set(available_backends()) == {"mock", "naive", "vllm"}


def test_unknown_backend_raises():
    config = _config()
    config.backend.name = "tensorrt"
    from llmbench.backends import BackendError

    with pytest.raises(BackendError):
        create_backend(config)


@pytest.mark.parametrize("profile", list(PROFILES))
def test_mock_profiles_all_work(profile):
    config = _config(gpu_profile=profile)
    backend = MockBackend(config)
    output = backend.generate(GenerationRequest(prompt="hello", max_new_tokens=8))
    assert output.success
    assert output.output_tokens == 8
    assert output.ttft_ms > 0
    assert output.e2e_ms >= output.ttft_ms


def test_mock_is_deterministic_without_noise():
    config = _config()
    backend = MockBackend(config)
    request = GenerationRequest(prompt="x" * 400, max_new_tokens=32, prompt_tokens_hint=256)
    first = backend.generate(request)
    second = backend.generate(request)
    assert first.ttft_ms == pytest.approx(second.ttft_ms)
    assert first.e2e_ms == pytest.approx(second.e2e_ms)


def test_tpot_matches_e2e_and_ttft():
    backend = MockBackend(_config())
    output = backend.generate(
        GenerationRequest(prompt="x", max_new_tokens=50, prompt_tokens_hint=128)
    )
    expected = (output.e2e_ms - output.ttft_ms) / (output.output_tokens - 1)
    assert output.tpot_ms == pytest.approx(expected)


def test_continuous_batching_beats_per_request_under_load():
    # Concurrency only shows up in the model if requests actually overlap, so
    # this test keeps a (tiny) non-zero wall-clock scale.
    continuous = MockBackend(_config("continuous_batch", time_scale=0.005))
    per_request = MockBackend(_config("per_request", time_scale=0.005))

    def aggregate(backend, n=8):
        """Simulated throughput: tokens / simulated makespan.

        Wall-clock throughput is meaningless here because the tests disable
        wall-clock sleeping (``time_scale=0``).
        """
        backend.reset_telemetry()
        requests = [
            GenerationRequest(
                prompt="x" * 300, request_id=f"r{i}", max_new_tokens=32, prompt_tokens_hint=256
            )
            for i in range(n)
        ]
        results = run_load(backend, requests, max_workers=n, arrival_mode="burst")
        tokens = sum(r["output_tokens"] for r in results.records)
        makespan_s = max(r["finish_ms"] for r in results.records) / 1000.0
        return tokens / makespan_s

    assert aggregate(continuous) > aggregate(per_request) * 1.5


def test_static_batch_returns_one_output_per_request():
    backend = MockBackend(_config("static_batch"))
    requests = [
        GenerationRequest(
            prompt="x" * (100 * (i + 1)),
            request_id=f"r{i}",
            max_new_tokens=16,
            prompt_tokens_hint=100 * (i + 1),
        )
        for i in range(4)
    ]
    outputs = backend.generate_batch(requests)
    assert len(outputs) == 4
    assert {o.request_id for o in outputs} == {"r0", "r1", "r2", "r3"}
    # Static batching finishes every request at the same time.
    assert len({o.e2e_ms for o in outputs}) == 1


def test_prefix_cache_reduces_ttft():
    config = _config("continuous_batch")
    config.backend.enable_prefix_caching = True
    backend = MockBackend(config)
    shared = "shared document " * 400

    first = backend.generate(
        GenerationRequest(
            prompt=shared + "q1",
            request_id="a",
            max_new_tokens=8,
            prompt_tokens_hint=2000,
            shared_prefix_tokens=1800,
        )
    )
    second = backend.generate(
        GenerationRequest(
            prompt=shared + "q1",
            request_id="b",
            max_new_tokens=8,
            prompt_tokens_hint=2000,
            shared_prefix_tokens=1800,
        )
    )
    assert second.ttft_ms < first.ttft_ms


def test_telemetry_reset_clears_accumulator():
    backend = MockBackend(_config())
    backend.generate(GenerationRequest(prompt="x", max_new_tokens=4, prompt_tokens_hint=64))
    assert backend.telemetry()["peak_memory_mb"] > 0
    backend.reset_telemetry()
    assert backend.telemetry() == {}


# ---------------------------------------------------------------------------
# Load generation
# ---------------------------------------------------------------------------
def test_burst_arrivals_are_all_zero():
    assert plan_arrivals(5, mode="burst") == [0.0] * 5


def test_constant_arrivals_are_uniform():
    offsets = plan_arrivals(5, mode="constant", rate_rps=10.0)
    assert offsets[0] == 0.0
    diffs = [b - a for a, b in zip(offsets, offsets[1:], strict=False)]
    assert all(d == pytest.approx(0.1) for d in diffs)


def test_poisson_arrivals_are_seeded():
    first = plan_arrivals(20, mode="poisson", rate_rps=5, seed=42)
    second = plan_arrivals(20, mode="poisson", rate_rps=5, seed=42)
    assert first == second
    assert all(b >= a for a, b in zip(first, first[1:], strict=False))


def test_run_load_records_all_fields():
    backend = MockBackend(_config())
    requests = [
        GenerationRequest(
            prompt="x" * 200,
            request_id=f"r{i}",
            max_new_tokens=16,
            prompt_tokens_hint=128,
            meta={"index": i},
        )
        for i in range(4)
    ]
    result = run_load(backend, requests, max_workers=2, arrival_mode="burst", concurrency=2)
    assert len(result.records) == 4
    record = result.records[0]
    for key in ("ttft_ms", "tpot_ms", "e2e_ms", "finish_ms", "queue_ms", "throughput_tok_s"):
        assert key in record
    assert record["concurrency"] == 2
    assert result.completed == 4


def test_run_load_survives_backend_exception():
    class Broken(MockBackend):
        def generate(self, request):  # noqa: ARG002
            raise RuntimeError("engine exploded")

    backend = Broken(_config())
    result = run_load(backend, [GenerationRequest(prompt="x", request_id="a")], max_workers=1)
    assert result.failed == 1
    assert result.records[0]["success"] is False
    assert "exploded" in result.records[0]["error"]


def test_run_batch_groups_requests():
    backend = MockBackend(_config("static_batch"))
    requests = [
        GenerationRequest(
            prompt="x" * 100, request_id=f"r{i}", max_new_tokens=8, prompt_tokens_hint=64
        )
        for i in range(5)
    ]
    result = run_batch(backend, requests, batch_size=2, concurrency=5)
    assert len(result.records) == 5
    assert result.records[0]["extra_batch_index"] == 0
    assert result.records[-1]["extra_batch_index"] == 2


def test_summarize_load_reports_percentiles():
    backend = MockBackend(_config())
    requests = [
        GenerationRequest(
            prompt="x" * 100, request_id=f"r{i}", max_new_tokens=8, prompt_tokens_hint=64
        )
        for i in range(6)
    ]
    summary = summarize_load(run_load(backend, requests, max_workers=3))
    assert summary["requests"] == 6
    assert summary["ttft_p95_ms"] >= summary["ttft_mean_ms"]


def test_empty_load_is_a_noop():
    backend = MockBackend(_config())
    assert run_load(backend, [], max_workers=2).records == []
    assert run_batch(backend, [], batch_size=2).records == []


def test_generation_output_defaults():
    output = GenerationOutput(request_id="x")
    assert output.success is True
    assert output.tpot_ms is None
    failed = output.failed("nope")
    assert failed.success is False and failed.error == "nope"
