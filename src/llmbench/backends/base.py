"""Backend abstraction shared by every inference engine.

The contract is deliberately narrow:

* ``load()`` / ``close()`` manage the lifecycle of the engine.
* ``generate(request)`` runs **one** request and returns a fully attributed
  :class:`GenerationOutput` (per-request TTFT, TPOT, end-to-end latency).
* ``generate_batch(requests)`` is optional and only implemented by engines
  that expose native static batching.

All timings are measured *inside* the backend, as close to the engine as
possible, which is why TTFT is a first-class field rather than something the
harness infers afterwards.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

from ..metrics.stats import safe_div

__all__ = ["GenerationRequest", "GenerationOutput", "Backend", "BackendError"]


class BackendError(RuntimeError):
    """Raised when an engine cannot be initialised or fails irrecoverably."""


@dataclass
class GenerationRequest:
    """A single inference request."""

    prompt: str
    request_id: str = "req-0"
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    #: Prompt length in tokens, when already known (avoids re-tokenising and
    #: keeps the mock backend deterministic).
    prompt_tokens_hint: int | None = None
    #: Number of leading tokens shared with previous requests in the same
    #: conversation (used by the prefix-caching scenario).
    shared_prefix_tokens: int = 0
    #: Free-form metadata propagated into result records.
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationOutput:
    """Result of a single request, with timings relative to request start."""

    request_id: str
    text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    ttft_ms: float | None = None
    e2e_ms: float = 0.0
    #: Timestamps (ms, relative to request start) at which each output token
    #: became available. Optional but valuable for TPOT and jitter analysis.
    token_times_ms: list[float] = field(default_factory=list)
    success: bool = True
    error: str | None = None
    backend: str | None = None
    serving_mode: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def tpot_ms(self) -> float | None:
        """Time per output token, excluding the prefill phase."""
        if self.ttft_ms is None or self.output_tokens <= 1:
            return None
        return safe_div(self.e2e_ms - self.ttft_ms, self.output_tokens - 1, None)  # type: ignore[arg-type]

    @property
    def throughput_tok_s(self) -> float | None:
        return safe_div(self.output_tokens * 1000.0, self.e2e_ms, None)  # type: ignore[arg-type]

    def failed(self, message: str) -> GenerationOutput:
        self.success = False
        self.error = message
        return self


class Backend(abc.ABC):
    """Base class for inference backends."""

    name: str = "base"
    serving_mode: str = "per_request"
    #: Whether :meth:`generate_batch` is a native batched execution path.
    supports_batch: bool = False

    def __init__(self, run_config):
        self.config = run_config
        self.model_config = run_config.model
        self.backend_config = run_config.backend
        self._loaded = False

    # -- lifecycle -------------------------------------------------------
    @abc.abstractmethod
    def load(self) -> None:
        """Initialise the engine. Must be idempotent."""

    @abc.abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationOutput:
        """Run a single request."""

    def generate_batch(self, requests: list[GenerationRequest]) -> list[GenerationOutput]:
        """Default batch implementation: sequential fallback."""
        return [self.generate(r) for r in requests]

    def close(self) -> None:
        """Release engine resources. Safe to call multiple times."""
        self._loaded = False

    # -- optional capabilities ------------------------------------------
    def telemetry(self) -> dict[str, float | None]:
        """Backend-reported resource metrics (used when no GPU is present)."""
        return {}

    def reset_telemetry(self) -> None:
        """Clear accumulated backend-reported metrics between measurements."""
        return

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "serving_mode": self.serving_mode,
            "supports_batch": self.supports_batch,
            "model": self.model_config.path,
        }

    def __enter__(self) -> Backend:
        if not self._loaded:
            self.load()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} name={self.name} mode={self.serving_mode}>"
