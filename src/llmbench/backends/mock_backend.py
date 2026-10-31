"""Deterministic, GPU-free backend used for demos, CI and smoke tests.

Why a mock backend?
-------------------
A benchmark harness is only useful if reviewers can run it. Most people who
clone the repository do not have a 24 GB GPU sitting idle, and CI certainly
does not. This backend implements an *analytic* performance model calibrated
against the reference RTX 3090 measurements shipped with this repository, so
the entire pipeline (load generation → telemetry → aggregation → figures →
dashboard) can be exercised end-to-end in seconds.

Every run produced by this backend is tagged ``is_mock=True`` in its metadata
so simulated numbers can never be mistaken for hardware measurements.

Model
-----
Concurrent requests are counted with an internal counter; aggregate GPU
throughput then grows sub-linearly with the in-flight batch::

    aggregate_decode(b) = decode_tok_s * min(max_speedup, b ** exponent)

The per-request rate is ``aggregate_decode(b) / b``, which reproduces the
signature result of continuous batching: **total throughput rises while
per-request latency stays roughly flat**. The exponent differs per serving
mode, which is what makes the three-way comparison informative:

===================  ==========  =========  ========
mode                 decode exp  max gain   KV/token
===================  ==========  =========  ========
continuous_batch         0.95      8.0×      0.19 MB
static_batch             0.80      5.5×      0.55 MB
per_request              0.30      2.0×      0.94 MB
===================  ==========  =========  ========
"""

from __future__ import annotations

import random
import threading
import time
import zlib
from dataclasses import dataclass, field

from ..core.logging import get_logger
from ..metrics.stats import safe_div
from .base import Backend, GenerationOutput, GenerationRequest

__all__ = ["MockBackend", "MockProfile", "ModeProfile", "PROFILES"]

log = get_logger("backends.mock")


@dataclass(frozen=True)
class ModeProfile:
    """Batching-efficiency parameters for one serving mode."""

    prefill_exponent: float
    prefill_max_speedup: float
    decode_exponent: float
    decode_max_speedup: float
    kv_mb_per_token: float
    util_at_one: float
    util_slope: float
    util_ceiling: float


@dataclass(frozen=True)
class MockProfile:
    """Hardware profile for the analytic simulator.

    Absolute rates are calibrated against the reference measurements in
    ``reports/benchmark_report.md`` (RTX 3090, Qwen2.5-7B-Instruct, fp16):
    39 tok/s single-stream decode for HF, 47 tok/s for vLLM, ~320 tok/s at
    concurrency 8.
    """

    name: str
    prefill_tok_s: float
    decode_tok_s: float
    idle_memory_mb: float
    modes: dict[str, ModeProfile] = field(default_factory=dict)

    def mode(self, name: str) -> ModeProfile:
        return self.modes.get(name) or self.modes["static_batch"]


def _mode_table(
    prefill_exp: float,
    decode_exp: float,
    decode_gain: float,
    kv: float,
    util_one: float,
    slope: float,
    ceiling: float,
) -> ModeProfile:
    return ModeProfile(
        prefill_exponent=prefill_exp,
        prefill_max_speedup=max(1.0, decode_gain * 0.5),
        decode_exponent=decode_exp,
        decode_max_speedup=decode_gain,
        kv_mb_per_token=kv,
        util_at_one=util_one,
        util_slope=slope,
        util_ceiling=ceiling,
    )


PROFILES: dict[str, MockProfile] = {
    "rtx3090": MockProfile(
        name="NVIDIA RTX 3090 24GB (simulated)",
        prefill_tok_s=3000.0,
        decode_tok_s=45.0,
        idle_memory_mb=14_200.0,
        modes={
            "continuous_batch": _mode_table(0.50, 0.95, 8.0, 0.19, 55.0, 2.2, 90.0),
            "static_batch": _mode_table(0.35, 0.80, 5.5, 0.55, 62.0, 3.0, 84.0),
            "per_request": _mode_table(0.25, 0.30, 2.0, 0.94, 45.0, 4.5, 78.0),
        },
    ),
    "a100": MockProfile(
        name="NVIDIA A100 80GB (simulated)",
        prefill_tok_s=6500.0,
        decode_tok_s=78.0,
        idle_memory_mb=14_400.0,
        modes={
            "continuous_batch": _mode_table(0.50, 0.95, 9.5, 0.19, 60.0, 1.6, 94.0),
            "static_batch": _mode_table(0.35, 0.80, 6.0, 0.55, 66.0, 2.4, 88.0),
            "per_request": _mode_table(0.25, 0.30, 2.2, 0.94, 48.0, 4.0, 80.0),
        },
    ),
    "h100": MockProfile(
        name="NVIDIA H100 80GB (simulated)",
        prefill_tok_s=11000.0,
        decode_tok_s=115.0,
        idle_memory_mb=14_400.0,
        modes={
            "continuous_batch": _mode_table(0.50, 0.94, 10.0, 0.19, 65.0, 1.2, 96.0),
            "static_batch": _mode_table(0.35, 0.80, 6.5, 0.55, 70.0, 2.0, 90.0),
            "per_request": _mode_table(0.25, 0.30, 2.3, 0.94, 52.0, 3.5, 82.0),
        },
    ),
    "cpu": MockProfile(
        name="CPU only (simulated)",
        prefill_tok_s=120.0,
        decode_tok_s=6.0,
        idle_memory_mb=1_200.0,
        modes={
            "continuous_batch": _mode_table(0.40, 0.50, 2.2, 0.19, 45.0, 4.0, 70.0),
            "static_batch": _mode_table(0.30, 0.45, 1.9, 0.55, 48.0, 5.0, 66.0),
            "per_request": _mode_table(0.20, 0.25, 1.5, 0.94, 35.0, 6.0, 55.0),
        },
    ),
}


def _estimate_tokens(text: str) -> int:
    """Cheap token estimate when no tokenizer is available.

    CJK characters are ~1 token each while Latin text is ~4 chars per token,
    so a blended heuristic of ``len / 1.8`` is a reasonable default.
    """
    if not text:
        return 0
    return max(1, int(len(text) / 1.8))


class MockBackend(Backend):
    """Analytic simulator implementing the :class:`Backend` contract."""

    name = "mock"

    def __init__(self, run_config):
        super().__init__(run_config)
        self.serving_mode = run_config.backend.effective_serving_mode
        if self.serving_mode in {"offline", "async"}:
            # Both map onto continuous batching for simulation purposes.
            self.serving_mode = "continuous_batch"
        profile_key = run_config.mock.gpu_profile
        if profile_key not in PROFILES:
            raise ValueError(
                f"Unknown mock GPU profile {profile_key!r}. " f"Available: {sorted(PROFILES)}"
            )
        self.profile = PROFILES[profile_key]
        self.mode = self.profile.mode(self.serving_mode)
        self.time_scale = run_config.mock.time_scale
        self.noise = run_config.mock.noise
        self._noise_seed = run_config.mock.seed

        self._lock = threading.Lock()
        self._active = 0
        self._max_active = 0
        self._telemetry: list[dict[str, float]] = []
        self._prefix_cache: set[str] = set()

    # -- lifecycle -------------------------------------------------------
    def load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        log.info(
            "Mock backend ready (profile=%s, mode=%s, time_scale=%.3f)",
            self.profile.name,
            self.serving_mode,
            self.time_scale,
        )

    def close(self) -> None:
        self._loaded = False

    # -- helpers ---------------------------------------------------------
    def _prompt_tokens(self, request: GenerationRequest) -> int:
        if request.prompt_tokens_hint:
            return int(request.prompt_tokens_hint)
        return _estimate_tokens(request.prompt)

    def _jitter(self, key: str) -> float:
        """Deterministic noise in ``[1-noise, 1+noise]``."""
        if self.noise <= 0:
            return 1.0
        rng = random.Random(zlib.crc32(key.encode("utf-8")) ^ self._noise_seed)
        return 1.0 + rng.uniform(-self.noise, self.noise)

    def _prefix_discount(self, request: GenerationRequest) -> float:
        """Fraction of the prompt that still needs prefill."""
        if not self.backend_config.enable_prefix_caching or self.serving_mode != "continuous_batch":
            return 1.0
        shared = min(request.shared_prefix_tokens, self._prompt_tokens(request))
        if shared <= 0:
            return 1.0
        with self._lock:
            hit = request.prompt[:64] in self._prefix_cache
            self._prefix_cache.add(request.prompt[:64])
        if not hit:
            return 1.0
        total = self._prompt_tokens(request)
        # A cache hit still pays block-table lookups (~4% of a full prefill).
        return max(0.04, 1.0 - shared / max(total, 1))

    def _simulate(
        self,
        prompt_tokens: int,
        output_tokens: int,
        batch: int,
        prefix_fraction: float,
        mode: str | None = None,
    ) -> dict[str, float]:
        """Return simulated latencies in **milliseconds**."""
        mode = mode or self.serving_mode
        p = self.profile
        m = p.mode(mode)
        batch = max(batch, 1)

        prefill_agg = p.prefill_tok_s * min(m.prefill_max_speedup, batch**m.prefill_exponent)
        decode_agg = p.decode_tok_s * min(m.decode_max_speedup, batch**m.decode_exponent)

        if mode == "static_batch":
            # Padding forces every sequence to the longest prompt in the batch,
            # and every request waits for the slowest one to finish.
            prefill_ms = (prompt_tokens * batch**0.25) / p.prefill_tok_s * 1000.0
            decode_ms = output_tokens * batch / decode_agg * 1000.0
        else:
            prefill_ms = (prompt_tokens * prefix_fraction) / prefill_agg * 1000.0
            decode_ms = output_tokens * batch / decode_agg * 1000.0

        return {
            "ttft_ms": prefill_ms,
            "decode_ms": decode_ms,
            "e2e_ms": prefill_ms + decode_ms,
        }

    def _record_telemetry(self, prompt_tokens: int, output_tokens: int, batch: int) -> None:
        p = self.profile
        m = self.mode
        kv_mb = m.kv_mb_per_token * (prompt_tokens + output_tokens)
        if self.serving_mode == "continuous_batch":
            util = m.util_ceiling - (m.util_ceiling - m.util_at_one) * (0.55**batch)
        else:
            util = max(m.util_ceiling * 0.6, m.util_at_one + 24.0 - m.util_slope * batch)
        with self._lock:
            self._telemetry.append(
                {
                    "peak_memory_mb": p.idle_memory_mb + kv_mb,
                    "kv_cache_delta_mb": kv_mb,
                    "avg_gpu_util_pct": util,
                }
            )

    # -- execution -------------------------------------------------------
    def generate(self, request: GenerationRequest) -> GenerationOutput:
        if not self._loaded:
            self.load()

        prompt_tokens = self._prompt_tokens(request)
        output_tokens = int(request.max_new_tokens)

        with self._lock:
            self._active += 1
            self._max_active = max(self._max_active, self._active)
            batch = self._active
        try:
            prefix_fraction = self._prefix_discount(request)
            sim = self._simulate(prompt_tokens, output_tokens, batch, prefix_fraction)
            factor = self._jitter(request.request_id)
            sim = {k: v * factor for k, v in sim.items()}

            self._sleep(sim["e2e_ms"])
            self._record_telemetry(prompt_tokens, output_tokens, batch)
        finally:
            with self._lock:
                self._active -= 1

        token_times = [
            sim["ttft_ms"] + i * (sim["decode_ms"] / max(output_tokens, 1))
            for i in range(output_tokens)
        ]
        return GenerationOutput(
            request_id=request.request_id,
            text=self._fake_text(request, output_tokens),
            input_tokens=prompt_tokens,
            output_tokens=output_tokens,
            ttft_ms=sim["ttft_ms"],
            e2e_ms=sim["e2e_ms"],
            token_times_ms=token_times,
            backend=self.name,
            serving_mode=self.serving_mode,
        )

    def generate_batch(self, requests: list[GenerationRequest]) -> list[GenerationOutput]:
        """Static batching: one padded forward, every request finishes together."""
        if not requests:
            return []
        if self.serving_mode != "static_batch":
            return super().generate_batch(requests)

        if not self._loaded:
            self.load()

        batch = len(requests)
        prompt_tokens = max(self._prompt_tokens(r) for r in requests)
        max_output = max(int(r.max_new_tokens) for r in requests)
        sim = self._simulate(prompt_tokens, max_output, batch, 1.0, mode="static_batch")
        factor = self._jitter("batch-" + ",".join(r.request_id for r in requests)[:64])
        sim = {k: v * factor for k, v in sim.items()}

        self._sleep(sim["e2e_ms"])
        self._record_telemetry(prompt_tokens, max_output, batch)

        outputs = []
        for request in requests:
            n = int(request.max_new_tokens)
            step = sim["decode_ms"] / max(max_output, 1)
            outputs.append(
                GenerationOutput(
                    request_id=request.request_id,
                    text=self._fake_text(request, n),
                    input_tokens=self._prompt_tokens(request),
                    output_tokens=n,
                    ttft_ms=sim["ttft_ms"],
                    e2e_ms=sim["e2e_ms"],
                    token_times_ms=[sim["ttft_ms"] + i * step for i in range(n)],
                    backend=self.name,
                    serving_mode=self.serving_mode,
                    extra={"batch_size": batch, "padded_prompt_tokens": prompt_tokens},
                )
            )
        return outputs

    # -- plumbing --------------------------------------------------------
    def _sleep(self, simulated_ms: float) -> None:
        """Emulate wall-clock time, scaled down by ``mock.time_scale``."""
        seconds = max(simulated_ms, 0.0) / 1000.0 * self.time_scale
        if seconds > 0:
            time.sleep(seconds)

    @staticmethod
    def _fake_text(request: GenerationRequest, output_tokens: int) -> str:
        snippet = request.prompt.strip().replace("\n", " ")[:48]
        return f"[mock output for {snippet!r}] " + ("token " * min(output_tokens, 8)).strip()

    def telemetry(self) -> dict[str, float | None]:
        with self._lock:
            samples = list(self._telemetry)
        if not samples:
            return {}
        return {
            "peak_memory_mb": max(s["peak_memory_mb"] for s in samples),
            "kv_cache_delta_mb": max(s["kv_cache_delta_mb"] for s in samples),
            "avg_gpu_util_pct": safe_div(
                sum(s["avg_gpu_util_pct"] for s in samples), len(samples), 0.0
            ),
            "max_gpu_util_pct": max(s["avg_gpu_util_pct"] for s in samples),
            "max_concurrency_observed": float(self._max_active),
        }

    def reset_telemetry(self) -> None:
        with self._lock:
            self._telemetry.clear()
            self._max_active = self._active
