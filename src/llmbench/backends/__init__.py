"""Backend registry.

``create_backend(config)`` is the single entry point used by the runner. Real
backends are imported lazily so that ``llmbench report`` works on a machine
without torch/vLLM installed.
"""

from __future__ import annotations

from collections.abc import Callable

from .base import Backend, BackendError, GenerationOutput, GenerationRequest

__all__ = [
    "Backend",
    "BackendError",
    "GenerationOutput",
    "GenerationRequest",
    "create_backend",
    "available_backends",
    "register_backend",
]

_REGISTRY: dict[str, Callable[[], type[Backend]]] = {}


def register_backend(name: str):
    """Decorator registering a backend factory under ``name``."""

    def _decorator(factory: Callable[[], type[Backend]]):
        _REGISTRY[name] = factory
        return factory

    return _decorator


def _mock_factory() -> type[Backend]:
    from .mock_backend import MockBackend

    return MockBackend


def _naive_factory() -> type[Backend]:
    from .naive_backend import NaiveBackend

    return NaiveBackend


def _vllm_factory() -> type[Backend]:
    from .vllm_backend import VLLMBackend

    return VLLMBackend


register_backend("mock")(_mock_factory)
register_backend("naive")(_naive_factory)
register_backend("vllm")(_vllm_factory)


def available_backends() -> dict[str, str]:
    """Return ``{name: short description}`` for every registered backend."""
    return {
        "mock": "Analytic simulator — no GPU required (results are synthetic)",
        "naive": "HuggingFace Transformers (per-request or static batching)",
        "vllm": "vLLM (continuous batching, PagedAttention)",
    }


def create_backend(config) -> Backend:
    """Instantiate the backend referenced by a :class:`RunConfig`."""
    name = config.backend.name
    if name not in _REGISTRY:
        raise BackendError(f"Unknown backend {name!r}. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]()(config)
