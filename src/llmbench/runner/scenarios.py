"""Benchmark scenarios.

Each scenario is a function ``run_x(config, backend, emit)`` that performs a
family of measurements and pushes result records through ``emit``.

Design rule: scenarios never compute aggregates. They emit *atomic*
measurements (one row per request, or one row per grid point) and let
``llmbench.core.results.aggregate_records`` derive p50/p95/p99 and
throughput. One source of truth, no drift between charts.
"""

from __future__ import annotations

from collections.abc import Callable

from ..backends.base import Backend, GenerationRequest
from ..core.config import RunConfig
from ..core.logging import get_logger
from .loadgen import run_batch, run_load
from .measure import Measurement
from .workload import build_workload, estimate_tokens

__all__ = [
    "run_single",
    "run_concurrency",
    "run_memory",
    "run_prefix_cache",
    "SHARED_DOCUMENT",
]

log = get_logger("runner.scenarios")

Emit = Callable[[dict], None]

SHARED_DOCUMENT = (
    "技术背景资料：大语言模型推理系统需要同时满足吞吐与延迟目标。"
    "PagedAttention 将 KV Cache 切分为固定大小的 block，通过块表映射实现按需分配，"
    "显存浪费从 60%~80% 降到 4% 以下。连续批处理在每次迭代时重新填充 batch，"
    "使新到达的请求无需等待当前批次结束。前缀缓存把共享前缀的 KV 结果持久化，"
    "在 RAG 与多轮对话中可直接命中。调度器需要同时考虑显存水位、序列长度分布与公平性，"
    "常见策略包括 FCFS、SJF 与带权重的优先级调度。可观测性指标包括首 token 延迟、"
    "单 token 延迟、端到端延迟与队列等待时间。量化可以降低权重与 KV Cache 的精度，"
    "在精度损失可控的前提下显著降低显存占用与带宽压力。张量并行把单层计算切分到多卡，"
    "流水线并行则按层切分，两者结合可以在多机上服务超大模型。投机解码使用小模型草稿"
    "再由大模型校验，在低温度场景下可获得 2~3 倍加速。"
) * 6


def _request(
    config: RunConfig,
    prompt: str,
    index: int,
    max_new_tokens: int | None = None,
    group: str | None = None,
    shared_prefix_tokens: int = 0,
    prompt_tokens_hint: int | None = None,
) -> GenerationRequest:
    return GenerationRequest(
        prompt=prompt,
        request_id=f"r{index:05d}",
        max_new_tokens=int(max_new_tokens or config.workload.max_new_tokens),
        temperature=config.workload.temperature,
        top_p=config.workload.top_p,
        prompt_tokens_hint=prompt_tokens_hint,
        shared_prefix_tokens=shared_prefix_tokens,
        meta={"index": index, "group": group},
    )


# ---------------------------------------------------------------------------
# 1. Single-request latency / breakdown
# ---------------------------------------------------------------------------
def run_single(config: RunConfig, backend: Backend, emit: Emit) -> None:
    """Sweep prompt length × output length with one request in flight.

    Produces the TTFT-vs-prompt-length curve and the prefill/decode split —
    measured, not derived from a proxy run.
    """
    idx = 0
    for prompt_len in config.scenarios.prompt_lengths:
        samples = build_workload(1, config.workload.dataset, config.seed, lengths=[prompt_len])
        prompt = samples[0].text
        for output_len in config.scenarios.output_lengths:
            for repeat in range(config.scenarios.repeats):
                request = _request(
                    config,
                    prompt,
                    idx,
                    max_new_tokens=output_len,
                    group=f"p{prompt_len}_o{output_len}",
                )
                with Measurement(config.monitor, backend) as measurement:
                    output = backend.generate(request)
                records = _to_records(
                    request,
                    output,
                    scenario="single",
                    concurrency=1,
                    repeat=repeat,
                    target_prompt_len=prompt_len,
                    requested_output_len=output_len,
                )
                measurement.attach(records)
                for record in records:
                    emit(record)
                log.info(
                    "single p=%d o=%d: ttft=%.1fms tpot=%.2fms e2e=%.0fms",
                    output.input_tokens,
                    output.output_tokens,
                    output.ttft_ms or float("nan"),
                    output.tpot_ms or float("nan"),
                    output.e2e_ms,
                )
                idx += 1


# ---------------------------------------------------------------------------
# 2. Concurrency / throughput
# ---------------------------------------------------------------------------
def run_concurrency(config: RunConfig, backend: Backend, emit: Emit) -> None:
    """Fan out requests at increasing concurrency and record every request."""
    arrival = config.workload.arrival
    for level in config.scenarios.concurrency_levels:
        workload = build_workload(level, config.workload.dataset, config.seed + level)
        requests = [
            _request(config, sample.text, i, prompt_tokens_hint=sample.actual_tokens)
            for i, sample in enumerate(workload)
        ]

        with Measurement(config.monitor, backend) as measurement:
            if backend.serving_mode == "static_batch" and backend.supports_batch:
                batch_size = min(level, config.backend.max_batch_size)
                result = run_batch(
                    backend,
                    requests,
                    batch_size=batch_size,
                    concurrency=level,
                )
            else:
                result = run_load(
                    backend,
                    requests,
                    max_workers=level,
                    arrival_mode=arrival.mode,
                    rate_rps=arrival.rate_rps,
                    seed=config.seed + level,
                    jitter=arrival.jitter,
                    concurrency=level,
                )
        measurement.attach(result.records)
        for record in result.records:
            record["scenario"] = "concurrency"
            record["bucket"] = "load"
            emit(record)

        log.info(
            "concurrency=%d: %.1f tok/s (%d ok, %d failed, wall %.2fs)",
            level,
            result.throughput_tok_s,
            result.completed,
            result.failed,
            result.wall_time_s,
        )


# ---------------------------------------------------------------------------
# 3. Memory vs sequence length
# ---------------------------------------------------------------------------
def run_memory(config: RunConfig, backend: Backend, emit: Emit) -> None:
    """Peak memory and KV-cache growth as prompt length increases."""
    output_len = 128
    for idx, prompt_len in enumerate(config.scenarios.prompt_lengths):
        samples = build_workload(1, config.workload.dataset, config.seed, lengths=[prompt_len])
        request = _request(
            config,
            samples[0].text,
            idx,
            max_new_tokens=output_len,
            group=f"mem_p{prompt_len}",
        )
        with Measurement(config.monitor, backend) as measurement:
            output = backend.generate(request)
        telemetry = measurement.telemetry
        emit(
            {
                "scenario": "memory",
                "level": "aggregate",
                "backend": output.backend,
                "serving_mode": output.serving_mode,
                "group": f"mem_p{prompt_len}",
                "bucket": f"mem_p{prompt_len}",
                "concurrency": 1,
                "prompt_len": output.input_tokens,
                "output_len": output.output_tokens,
                "input_tokens": output.input_tokens,
                "output_tokens": output.output_tokens,
                "ttft_ms": output.ttft_ms,
                "e2e_ms": output.e2e_ms,
                "peak_memory_mb": telemetry.peak_memory_mb,
                "kv_cache_delta_mb": telemetry.kv_cache_delta_mb,
                "avg_gpu_util_pct": telemetry.avg_gpu_util_pct,
                "success": output.success,
                "error": output.error,
                "extra_telemetry_source": telemetry.source,
            }
        )
        log.info(
            "memory p=%d: peak=%.0fMB kv_delta=%.0fMB",
            output.input_tokens,
            telemetry.peak_memory_mb or 0.0,
            telemetry.kv_cache_delta_mb or 0.0,
        )


# ---------------------------------------------------------------------------
# 4. Prefix caching (multi-turn, shared document)
# ---------------------------------------------------------------------------
def run_prefix_cache(
    config: RunConfig,
    backend: Backend,
    emit: Emit,
    cache_enabled: bool,
    rounds: int = 3,
) -> None:
    """Multi-turn conversation over a long shared document.

    Each round re-sends the full history, so rounds 2..N share a prefix with
    the previous request. With prefix caching enabled the shared KV blocks are
    reused and TTFT should drop; without it every round re-prefills everything.
    """
    if backend.serving_mode == "static_batch" and backend.name == "naive":
        log.info("Prefix caching is not supported by the HuggingFace backend — skipping.")
        return

    group = "cache_on" if cache_enabled else "cache_off"
    assistant_replies: list[str] = []

    questions = [
        "请用三句话概括这份资料的核心结论。",
        "其中哪一种机制对降低首 token 延迟最有效？为什么？",
        "如果要把吞吐再提升一倍，你会优先调整哪些参数？",
    ]

    prefix_tokens = estimate_tokens(SHARED_DOCUMENT)
    for round_idx in range(min(rounds, len(questions))):
        # Rebuild the conversation transcript with the previous answers.
        parts = [SHARED_DOCUMENT]
        for q, a in zip(questions[:round_idx], assistant_replies, strict=False):
            parts.append(f"问：{q}\n答：{a}")
        parts.append(f"问：{questions[round_idx]}\n答：")
        prompt = "\n\n".join(parts)

        request = _request(
            config,
            prompt,
            round_idx,
            max_new_tokens=min(config.workload.max_new_tokens, 96),
            group=group,
            shared_prefix_tokens=prefix_tokens,
        )
        with Measurement(config.monitor, backend) as measurement:
            output = backend.generate(request)
        assistant_replies.append(output.text[:200])

        records = _to_records(
            request,
            output,
            scenario="prefix_cache",
            concurrency=1,
            repeat=0,
            target_prompt_len=None,
            requested_output_len=request.max_new_tokens,
            extra={"round": round_idx + 1, "cache_enabled": cache_enabled},
        )
        measurement.attach(records)
        for record in records:
            record["group"] = f"{group}_r{round_idx + 1}"
            record["bucket"] = f"{group}_r{round_idx + 1}"
            emit(record)
        log.info(
            "prefix_cache %s round=%d: ttft=%.1fms e2e=%.0fms",
            group,
            round_idx + 1,
            output.ttft_ms or float("nan"),
            output.e2e_ms,
        )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _to_records(
    request: GenerationRequest,
    output,
    scenario: str,
    concurrency: int | None,
    repeat: int,
    target_prompt_len: int | None,
    requested_output_len: int | None,
    extra: dict | None = None,
) -> list[dict]:
    tpot = output.tpot_ms
    record = {
        "scenario": scenario,
        "level": "request",
        "backend": output.backend or request.meta.get("backend"),
        "serving_mode": output.serving_mode,
        "group": request.meta.get("group"),
        "bucket": request.meta.get("group") or scenario,
        "repeat": repeat,
        "request_index": request.meta.get("index"),
        "concurrency": concurrency,
        "prompt_len": output.input_tokens,
        "output_len": output.output_tokens,
        "input_tokens": output.input_tokens,
        "output_tokens": output.output_tokens,
        "arrival_ms": 0.0,
        "start_ms": 0.0,
        "queue_ms": 0.0,
        "ttft_ms": output.ttft_ms,
        "tpot_ms": tpot,
        "e2e_ms": output.e2e_ms,
        "finish_ms": output.e2e_ms,
        "total_time_ms": output.e2e_ms,
        "throughput_tok_s": output.throughput_tok_s,
        "success": output.success,
        "error": output.error,
        "target_prompt_len": target_prompt_len,
        "requested_output_len": requested_output_len,
    }
    for key, value in (extra or {}).items():
        record[key] = value
    for key, value in (output.extra or {}).items():
        record[f"extra_{key}"] = value
    return [record]
