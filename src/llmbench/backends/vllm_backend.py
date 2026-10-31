"""vLLM backend.

Two execution paths are provided because vLLM exposes two very different
entry points:

``continuous_batch``
    Uses the *async* engine (``AsyncLLM`` on vLLM ≥ 0.7, ``AsyncLLMEngine``
    on older releases) and streams tokens. Every request is submitted
    independently, so the numbers include real scheduler queueing and real
    per-request TTFT — the only way to measure continuous batching honestly.

``offline``
    Uses the offline ``LLM.generate`` batch API. Faster to set up, but all
    requests are returned together, so per-request TTFT is only available
    when the installed vLLM version exposes ``RequestOutput.metrics``.

The backend degrades gracefully: if the streaming path cannot be imported it
logs a warning and falls back to the offline path.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from concurrent.futures import Future
from typing import Any

from ..core.logging import get_logger
from .base import Backend, BackendError, GenerationOutput, GenerationRequest

__all__ = ["VLLMBackend"]

log = get_logger("backends.vllm")


def _import_vllm():
    try:
        import vllm  # noqa: PLC0415

        return vllm
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise BackendError(
            "The 'vllm' package is required for the vLLM backend.\n"
            "Install the GPU extras first:  pip install -r requirements-gpu.txt"
        ) from exc


class _LoopRunner:
    """A dedicated asyncio event loop running in a background thread."""

    def __init__(self, name: str = "vllm-loop"):
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._ready = threading.Event()
        self._thread.start()
        self._ready.wait(timeout=10)

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.call_soon(self._ready.set)
        try:
            self._loop.run_forever()
        finally:
            pending = asyncio.all_tasks(self._loop)
            for task in pending:
                task.cancel()
            if pending:
                self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            self._loop.close()

    def run_sync(self, fn, timeout: float | None = None):
        """Execute a synchronous callable inside the loop thread."""
        future: Future = Future()

        def _callback() -> None:
            try:
                future.set_result(fn())
            except BaseException as exc:  # noqa: BLE001 - propagate to caller
                future.set_exception(exc)

        self._loop.call_soon_threadsafe(_callback)
        return future.result(timeout=timeout)

    def submit(self, coro, timeout: float | None = None):
        cfuture = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return cfuture.result(timeout=timeout)

    def stop(self) -> None:
        if self._loop.is_closed():
            return
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=10)


class VLLMBackend(Backend):
    """vLLM engine wrapper with per-request timing."""

    name = "vllm"
    supports_batch = True

    def __init__(self, run_config):
        super().__init__(run_config)
        requested = run_config.backend.effective_serving_mode
        self.serving_mode = "offline" if requested in {"offline"} else "continuous_batch"
        self._llm = None
        self._engine = None
        self._engine_kind: str | None = None
        self._loop: _LoopRunner | None = None
        self._tokenizer = None
        self._request_prefix = f"req-{int(time.time() * 1000) % 10_000_000}"
        self._timeout = run_config.workload.request_timeout_s

    # -- lifecycle -------------------------------------------------------
    def _engine_kwargs(self) -> dict[str, Any]:
        cfg = self.backend_config
        kwargs: dict[str, Any] = {
            "model": self.model_config.resolve_path(),
            "tensor_parallel_size": cfg.tensor_parallel_size,
            "gpu_memory_utilization": cfg.gpu_memory_utilization,
            "max_model_len": self.model_config.max_model_len,
            "dtype": self.model_config.dtype,
            "trust_remote_code": self.model_config.trust_remote_code,
            "enforce_eager": cfg.enforce_eager,
            "max_num_seqs": cfg.max_num_seqs,
            "disable_log_stats": True,
        }
        if cfg.enable_prefix_caching:
            kwargs["enable_prefix_caching"] = True
        if self.model_config.quantization:
            kwargs["quantization"] = self.model_config.quantization
        kwargs.update(cfg.extra or {})
        return kwargs

    def load(self) -> None:
        if self._loaded:
            return
        _import_vllm()

        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.resolve_tokenizer_path(),
            trust_remote_code=self.model_config.trust_remote_code,
            local_files_only=True,
        )

        if self.serving_mode == "offline":
            self._load_offline()
        else:
            try:
                self._load_async()
            except Exception as exc:  # pragma: no cover - version dependent
                log.warning(
                    "Streaming engine unavailable (%s). Falling back to the offline "
                    "batch API; per-request TTFT will be approximated.",
                    exc,
                )
                self._load_offline()
        self._loaded = True

    def _load_offline(self) -> None:
        vllm = _import_vllm()
        self.serving_mode = "offline"
        log.info("Initialising vLLM offline engine (this can take a minute)…")
        self._llm = vllm.LLM(**self._engine_kwargs())
        log.info("vLLM offline engine ready.")

    def _load_async(self) -> None:
        from vllm.engine.arg_utils import AsyncEngineArgs

        self._loop = _LoopRunner()
        args = AsyncEngineArgs(**self._engine_kwargs())
        log.info("Initialising vLLM async engine (this can take a minute)…")

        def _create():
            try:
                from vllm.v1.engine.async_llm import AsyncLLM  # vLLM >= 0.7

                self._engine_kind = "v1"
                return AsyncLLM.from_engine_args(args)
            except Exception:
                from vllm.engine.async_llm_engine import AsyncLLMEngine

                self._engine_kind = "legacy"
                return AsyncLLMEngine.from_engine_args(args)

        self._engine = self._loop.run_sync(_create, timeout=900)
        log.info("vLLM async engine ready (api=%s).", self._engine_kind)

    def close(self) -> None:
        engine, llm, loop = self._engine, self._llm, self._loop
        self._engine = None
        self._llm = None
        self._loop = None
        self._loaded = False

        if engine is not None:
            try:
                for name in ("shutdown", "abort_all"):
                    fn = getattr(engine, name, None)
                    if callable(fn) and loop is not None:
                        loop.run_sync(fn, timeout=30)
                        break
            except Exception as exc:  # pragma: no cover
                log.debug("Engine shutdown warning: %s", exc)
        if loop is not None:
            with contextlib.suppress(Exception):
                loop.stop()
        del engine, llm

        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover
            pass

    # -- helpers ---------------------------------------------------------
    def _format_prompt(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        try:
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:  # pragma: no cover
            return prompt

    def _sampling_params(self, request: GenerationRequest):
        from vllm import SamplingParams

        greedy = request.temperature <= 0
        return SamplingParams(
            temperature=0.0 if greedy else request.temperature,
            top_p=1.0 if greedy else request.top_p,
            max_tokens=int(request.max_new_tokens),
        )

    @staticmethod
    def _ttft_from_metrics(output) -> float | None:
        """Derive TTFT from ``RequestOutput.metrics`` when available."""
        metrics = getattr(output, "metrics", None)
        if metrics is None:
            return None
        arrival = getattr(metrics, "arrival_time", None)
        first = getattr(metrics, "first_token_time", None)
        if arrival is None or first is None:
            return None
        delta = (first - arrival) * 1000.0
        return delta if delta >= 0 else None

    # -- execution -------------------------------------------------------
    def generate(self, request: GenerationRequest) -> GenerationOutput:
        if not self._loaded:
            self.load()
        if self.serving_mode == "offline":
            return self._generate_offline(request)
        return self._generate_streaming(request)

    def _generate_offline(self, request: GenerationRequest) -> GenerationOutput:
        prompt = self._format_prompt(request.prompt)
        params = self._sampling_params(request)

        t0 = time.perf_counter()
        outputs = self._llm.generate([prompt], params)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if not outputs or not outputs[0].outputs:
            return GenerationOutput(request_id=request.request_id, e2e_ms=elapsed_ms).failed(
                "vLLM returned no output"
            )

        out = outputs[0]
        text = out.outputs[0].text
        output_tokens = len(out.outputs[0].token_ids)
        ttft_ms = self._ttft_from_metrics(out)

        return GenerationOutput(
            request_id=request.request_id,
            text=text,
            input_tokens=len(out.prompt_token_ids),
            output_tokens=output_tokens,
            ttft_ms=ttft_ms if ttft_ms is not None else elapsed_ms,
            e2e_ms=elapsed_ms,
            backend=self.name,
            serving_mode="offline",
            extra={"ttft_source": "metrics" if ttft_ms is not None else "e2e_proxy"},
        )

    def _generate_streaming(self, request: GenerationRequest) -> GenerationOutput:
        prompt = self._format_prompt(request.prompt)
        coro = self._agenerate(request, prompt)
        assert self._loop is not None, "streaming engine not initialised"
        try:
            return self._loop.submit(coro, timeout=self._timeout)
        except Exception as exc:  # pragma: no cover - engine failure
            log.error("vLLM streaming request %s failed: %s", request.request_id, exc)
            return GenerationOutput(request_id=request.request_id).failed(str(exc))

    async def _agenerate(self, request: GenerationRequest, prompt: str) -> GenerationOutput:
        params = self._sampling_params(request)
        request_id = f"{self._request_prefix}-{request.request_id}"

        t0 = time.perf_counter()
        first_token_at: float | None = None
        token_times: list[float] = []
        last_output = None
        seen_tokens = 0

        async for output in self._engine.generate(prompt, params, request_id):
            now = time.perf_counter()
            last_output = output
            for choice in output.outputs:
                produced = len(choice.token_ids)
                if produced > seen_tokens:
                    # Tokens typically arrive in chunks (streaming interval);
                    # spread them evenly inside the chunk interval.
                    start = token_times[-1] if token_times else (now - t0) * 1000.0
                    end = (now - t0) * 1000.0
                    for i in range(produced - seen_tokens):
                        frac = (i + 1) / (produced - seen_tokens)
                        token_times.append(start + (end - start) * frac)
                    seen_tokens = produced
            if first_token_at is None and seen_tokens > 0:
                first_token_at = now

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if last_output is None or not last_output.outputs:
            return GenerationOutput(request_id=request.request_id, e2e_ms=elapsed_ms).failed(
                "vLLM stream produced no output"
            )

        ttft_ms = (first_token_at - t0) * 1000.0 if first_token_at else None
        if ttft_ms is None:
            ttft_ms = self._ttft_from_metrics(last_output) or elapsed_ms

        return GenerationOutput(
            request_id=request.request_id,
            text=last_output.outputs[0].text,
            input_tokens=len(last_output.prompt_token_ids or []),
            output_tokens=len(last_output.outputs[0].token_ids),
            ttft_ms=ttft_ms,
            e2e_ms=elapsed_ms,
            token_times_ms=token_times,
            backend=self.name,
            serving_mode="continuous_batch",
            extra={"engine_api": self._engine_kind},
        )

    def generate_batch(self, requests: list[GenerationRequest]) -> list[GenerationOutput]:
        if not requests:
            return []
        if self.serving_mode != "offline":
            # With a live async engine, "a batch" simply means: in-flight
            # requests share the scheduler — which is what the load generator
            # exercises anyway.
            return [self._generate_streaming(r) for r in requests]
        if not self._loaded:
            self.load()

        prompts = [self._format_prompt(r.prompt) for r in requests]
        params = self._sampling_params(requests[0])
        t0 = time.perf_counter()
        outputs = self._llm.generate(prompts, params)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        results: list[GenerationOutput] = []
        for request, out in zip(requests, outputs, strict=False):
            ttft_ms = self._ttft_from_metrics(out)
            results.append(
                GenerationOutput(
                    request_id=request.request_id,
                    text=out.outputs[0].text,
                    input_tokens=len(out.prompt_token_ids),
                    output_tokens=len(out.outputs[0].token_ids),
                    ttft_ms=ttft_ms if ttft_ms is not None else elapsed_ms,
                    e2e_ms=elapsed_ms,
                    backend=self.name,
                    serving_mode="offline",
                    extra={"batch_size": len(requests)},
                )
            )
        return results
