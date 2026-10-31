"""Load generation with real arrival processes.

A benchmark that submits everything at once measures *throughput ceiling*, not
*serving behaviour*. This module schedules requests over time (burst, constant
rate or Poisson), dispatches them from a bounded worker pool, and records for
each request:

* ``arrival_ms``  — when the request *should* have been sent
* ``start_ms``    — when the backend actually started it (queueing delay)
* ``ttft_ms``     — first token (measured inside the backend)
* ``e2e_ms``      — end of stream
* ``finish_ms``   — ``start_ms + e2e_ms`` (used for makespan / timeline plots)
"""

from __future__ import annotations

import math
import random
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from ..backends.base import Backend, GenerationOutput, GenerationRequest
from ..core.logging import get_logger
from ..metrics.stats import safe_div

__all__ = ["LoadRun", "plan_arrivals", "run_load", "run_batch"]

log = get_logger("runner.loadgen")


@dataclass
class LoadRun:
    """Outcome of dispatching a set of requests."""

    records: list[dict] = field(default_factory=list)
    wall_time_s: float = 0.0
    submitted: int = 0
    completed: int = 0
    failed: int = 0
    total_output_tokens: int = 0

    @property
    def throughput_tok_s(self) -> float:
        return safe_div(self.total_output_tokens, self.wall_time_s, 0.0)


def plan_arrivals(
    n: int,
    mode: str = "burst",
    rate_rps: float = 8.0,
    seed: int = 42,
    jitter: float = 0.0,
) -> list[float]:
    """Return arrival offsets in seconds for ``n`` requests."""
    if n <= 0:
        return []
    if mode == "burst":
        return [0.0] * n

    rng = random.Random(seed)
    interval = 1.0 / max(rate_rps, 1e-6)
    offsets: list[float] = []
    cursor = 0.0
    for _ in range(n):
        gap = rng.expovariate(rate_rps) if mode == "poisson" else interval
        if jitter:
            gap *= 1.0 + rng.uniform(-jitter, jitter)
        cursor += gap
        offsets.append(cursor)
    # Normalise so the first request arrives at t=0.
    base = offsets[0] if offsets else 0.0
    return [max(0.0, o - base) for o in offsets]


def _make_record(
    request: GenerationRequest,
    output: GenerationOutput,
    arrival_ms: float,
    start_ms: float,
    scenario: str,
    concurrency: int | None,
    repeat: int,
    prompt_len: int | None = None,
) -> dict:
    ttft = output.ttft_ms
    tpot = output.tpot_ms
    finish_ms = start_ms + output.e2e_ms
    return {
        "scenario": scenario,
        "level": "request",
        "backend": output.backend,
        "serving_mode": output.serving_mode,
        "group": request.meta.get("group"),
        "repeat": repeat,
        "request_index": request.meta.get("index"),
        "concurrency": concurrency,
        "prompt_len": prompt_len if prompt_len is not None else output.input_tokens,
        "output_len": output.output_tokens,
        "input_tokens": output.input_tokens,
        "output_tokens": output.output_tokens,
        "arrival_ms": round(arrival_ms, 3),
        "start_ms": round(start_ms, 3),
        "queue_ms": round(max(start_ms - arrival_ms, 0.0), 3),
        "ttft_ms": round(ttft, 3) if ttft is not None else None,
        "tpot_ms": round(tpot, 4) if tpot is not None else None,
        "e2e_ms": round(output.e2e_ms, 3),
        "finish_ms": round(finish_ms, 3),
        "total_time_ms": round(output.e2e_ms, 3),
        "throughput_tok_s": round(output.throughput_tok_s or 0.0, 3),
        "success": output.success,
        "error": output.error,
        **{f"extra_{k}": v for k, v in (output.extra or {}).items()},
    }


def run_load(
    backend: Backend,
    requests: Sequence[GenerationRequest],
    max_workers: int = 8,
    arrival_mode: str = "burst",
    rate_rps: float = 8.0,
    seed: int = 42,
    jitter: float = 0.0,
    scenario: str = "concurrency",
    concurrency: int | None = None,
    repeat: int = 0,
    on_progress: Callable[[int, int], None] | None = None,
) -> LoadRun:
    """Dispatch requests according to an arrival schedule.

    ``max_workers`` bounds the number of in-flight requests, i.e. it *is* the
    concurrency level. Requests beyond that queue in the executor, which is
    exactly what a real server's admission queue does.
    """
    if not requests:
        return LoadRun()

    concurrency = concurrency or max_workers
    offsets = plan_arrivals(len(requests), arrival_mode, rate_rps, seed, jitter)
    result = LoadRun(submitted=len(requests))
    write_lock = threading.Lock()

    def _submit(
        executor: ThreadPoolExecutor, request: GenerationRequest, offset_s: float
    ) -> Future:
        delay = offset_s - (time.perf_counter() - t_start)
        if delay > 0:
            time.sleep(delay)
        arrived_ms = (time.perf_counter() - t_start) * 1000.0
        future = executor.submit(_invoke, request, arrived_ms)
        request.meta["arrival_ms"] = arrived_ms
        return future

    def _invoke(request: GenerationRequest, arrived_ms: float):
        start = time.perf_counter()
        try:
            output = backend.generate(request)
        except Exception as exc:  # noqa: BLE001 - benchmark must survive failures
            log.error("Request %s failed: %s", request.request_id, exc)
            output = GenerationOutput(request_id=request.request_id).failed(str(exc))
            output.backend = backend.name
            output.serving_mode = backend.serving_mode
        start_ms = (start - t_start) * 1000.0
        return output, arrived_ms, start_ms

    t_start = time.perf_counter()
    futures: dict[Future, GenerationRequest] = {}
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="loadgen") as pool:
        for request, offset in zip(requests, offsets, strict=False):
            futures[_submit(pool, request, offset)] = request

        for done, future in enumerate(as_completed(futures), start=1):
            request = futures[future]
            try:
                output, arrived_ms, start_ms = future.result()
            except Exception as exc:  # pragma: no cover - defensive
                output = GenerationOutput(request_id=request.request_id).failed(str(exc))
                arrived_ms = request.meta.get("arrival_ms", 0.0)
                start_ms = arrived_ms

            record = _make_record(
                request,
                output,
                arrival_ms=arrived_ms,
                start_ms=start_ms,
                scenario=scenario,
                concurrency=concurrency,
                repeat=repeat,
            )
            with write_lock:
                result.records.append(record)
                result.total_output_tokens += output.output_tokens if output.success else 0
                if output.success:
                    result.completed += 1
                else:
                    result.failed += 1
            if on_progress:
                on_progress(done, len(requests))

    result.wall_time_s = time.perf_counter() - t_start
    result.records.sort(key=lambda r: (r.get("request_index") is None, r.get("request_index")))
    return result


def run_batch(
    backend: Backend,
    requests: Sequence[GenerationRequest],
    batch_size: int,
    scenario: str = "concurrency",
    concurrency: int | None = None,
    repeat: int = 0,
) -> LoadRun:
    """Execute requests as static batches of ``batch_size``.

    Used for engines whose only batched path is "pad everything to the longest
    sequence and run one forward pass". Each batch is submitted as a unit, so
    every request in a batch shares the same TTFT and completion time — the
    behaviour static batching exhibits in production.
    """
    result = LoadRun(submitted=len(requests))
    if not requests:
        return result

    concurrency = concurrency or batch_size
    raw_start = time.perf_counter()
    for batch_index, offset in enumerate(range(0, len(requests), batch_size)):
        chunk = list(requests[offset : offset + batch_size])
        arrived_ms = (time.perf_counter() - raw_start) * 1000.0
        t0 = time.perf_counter()
        try:
            outputs = backend.generate_batch(chunk)
        except Exception as exc:  # noqa: BLE001
            log.error("Batch %d failed: %s", batch_index, exc)
            outputs = [GenerationOutput(request_id=r.request_id).failed(str(exc)) for r in chunk]
        start_ms = (t0 - raw_start) * 1000.0

        for request, output in zip(chunk, outputs, strict=False):
            output.backend = output.backend or backend.name
            output.serving_mode = output.serving_mode or backend.serving_mode
            record = _make_record(
                request,
                output,
                arrival_ms=arrived_ms,
                start_ms=start_ms,
                scenario=scenario,
                concurrency=concurrency,
                repeat=repeat,
            )
            record["extra_batch_index"] = batch_index
            result.records.append(record)
            result.total_output_tokens += output.output_tokens if output.success else 0
            if output.success:
                result.completed += 1
            else:
                result.failed += 1

    result.wall_time_s = time.perf_counter() - raw_start
    return result


def summarize_load(run: LoadRun) -> dict[str, float]:
    """Aggregate a :class:`LoadRun` into headline metrics."""
    from ..metrics.stats import percentile

    ttfts = [r["ttft_ms"] for r in run.records if r.get("ttft_ms") is not None]
    e2es = [r["e2e_ms"] for r in run.records if r.get("e2e_ms") is not None]
    queue = [r["queue_ms"] for r in run.records if r.get("queue_ms") is not None]
    return {
        "throughput_tok_s": run.throughput_tok_s,
        "wall_time_s": run.wall_time_s,
        "requests": len(run.records),
        "completed": run.completed,
        "failed": run.failed,
        "ttft_mean_ms": sum(ttfts) / len(ttfts) if ttfts else math.nan,
        "ttft_p95_ms": percentile(ttfts, 95) or math.nan,
        "e2e_mean_ms": sum(e2es) / len(e2es) if e2es else math.nan,
        "e2e_p95_ms": percentile(e2es, 95) or math.nan,
        "queue_mean_ms": sum(queue) / len(queue) if queue else math.nan,
    }
