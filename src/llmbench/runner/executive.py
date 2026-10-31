"""Suite orchestration: build a backend, run the selected scenarios, persist.

The executive owns *process-level* concerns so that individual scenarios stay
pure:

* engine lifecycle and warm-up
* GPU memory release between engines (vLLM in particular holds memory until
  the process exits unless it is closed explicitly)
* metadata capture and result persistence
* a single place where a failure is recorded as a failed run instead of a
  traceback
"""

from __future__ import annotations

import contextlib
import dataclasses
import gc
import time
from collections.abc import Callable, Iterable
from pathlib import Path

from ..backends import create_backend
from ..backends.base import Backend, GenerationRequest
from ..core.config import RunConfig
from ..core.logging import get_logger
from ..core.results import RunMetadata, RunWriter, build_metadata
from ..metrics.system import collect_system_info
from . import scenarios as scenario_impl

__all__ = ["SCENARIOS", "selected_scenarios", "run_suite", "release_gpu"]

log = get_logger("runner.executive")

SCENARIOS = ("single", "concurrency", "memory", "prefix_cache", "scheduler")

#: Scenarios that need a warmed-up inference engine.
_ENGINE_SCENARIOS = {"single", "concurrency", "memory", "prefix_cache"}


def selected_scenarios(config: RunConfig) -> list[str]:
    flags = {
        "single": config.scenarios.single,
        "concurrency": config.scenarios.concurrency,
        "memory": config.scenarios.memory,
        "prefix_cache": config.scenarios.prefix_cache,
        "scheduler": config.scenarios.scheduler,
    }
    return [name for name in SCENARIOS if flags.get(name, False)]


def release_gpu(wait_s: float = 0.5) -> None:
    """Best-effort GPU memory release between engines."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass
    if wait_s:
        time.sleep(wait_s)


def _warmup(backend: Backend, config: RunConfig) -> None:
    runs = config.workload.warmup_runs
    if runs <= 0:
        return
    log.info("Warming up %s (%d run(s))…", backend.name, runs)
    for i in range(runs):
        request = GenerationRequest(
            prompt="你好，请用一句话介绍你自己。",
            request_id=f"warmup-{i}",
            max_new_tokens=16,
            temperature=0.0,
        )
        try:
            backend.generate(request)
        except Exception as exc:  # pragma: no cover - warm-up is best effort
            log.warning("Warm-up request failed: %s", exc)
            break


def _with_cache_flag(config: RunConfig, enabled: bool) -> RunConfig:
    return dataclasses.replace(
        config, backend=dataclasses.replace(config.backend, enable_prefix_caching=enabled)
    )


def run_suite(
    config: RunConfig,
    scenarios: Iterable[str] | None = None,
    metadata: RunMetadata | None = None,
    on_scenario: Callable[[str], None] | None = None,
) -> Path:
    """Execute the configured scenarios and return the run directory."""
    names = list(scenarios) if scenarios is not None else selected_scenarios(config)
    unknown = [n for n in names if n not in SCENARIOS]
    if unknown:
        raise ValueError(f"Unknown scenario(s) {unknown}; choose from {list(SCENARIOS)}")
    if not names:
        raise ValueError("No scenarios selected — enable at least one in `scenarios:`")

    system = collect_system_info(mock=config.backend.name == "mock")
    if config.backend.name == "mock":
        # Be explicit: the reported "GPU" is the simulated target, not the host.
        from ..backends.mock_backend import PROFILES

        profile = PROFILES.get(config.mock.gpu_profile)
        system["gpu_name"] = profile.name if profile else config.mock.gpu_profile
        system["simulated_profile"] = config.mock.gpu_profile
        system["mock_time_scale"] = config.mock.time_scale
    meta = metadata or build_metadata(config, scenarios=names, system=system)
    meta.scenarios = names
    if not meta.system:
        meta.system = system

    writer = RunWriter(meta)
    backend: Backend | None = None
    started = time.perf_counter()

    def emit(record: dict) -> None:
        writer.add(record)

    try:
        engine_names = [n for n in names if n in _ENGINE_SCENARIOS and n != "prefix_cache"]
        if engine_names:
            backend = create_backend(config)
            backend.load()
            _warmup(backend, config)
            for name in engine_names:
                if on_scenario:
                    on_scenario(name)
                log.info("Running scenario: %s", name)
                _dispatch(name, config, backend, emit)
            backend.close()
            backend = None
            release_gpu()

        if "prefix_cache" in names:
            _run_prefix_cache_scenarios(config, names, emit, on_scenario)

        if "scheduler" in names:
            if on_scenario:
                on_scenario("scheduler")
            log.info("Running scenario: scheduler (simulation)")
            from ..scheduler_sim.runner import run_scheduler_comparison

            for record in run_scheduler_comparison(config):
                emit(record)

        meta.notes = f"wall_time={time.perf_counter() - started:.1f}s"
        path = writer.write()
        log.info("Run saved to %s", path)
        return path
    except BaseException as exc:
        if backend is not None:
            with contextlib.suppress(Exception):
                backend.close()
        writer.fail(f"{type(exc).__name__}: {exc}")
        log.error("Run failed: %s", exc)
        raise
    finally:
        release_gpu(wait_s=0.0)


def _dispatch(name: str, config: RunConfig, backend: Backend, emit: Callable[[dict], None]) -> None:
    if name == "single":
        scenario_impl.run_single(config, backend, emit)
    elif name == "concurrency":
        scenario_impl.run_concurrency(config, backend, emit)
    elif name == "memory":
        scenario_impl.run_memory(config, backend, emit)
    else:  # pragma: no cover - guarded above
        raise ValueError(f"Unhandled scenario {name}")


def _run_prefix_cache_scenarios(
    config: RunConfig,
    names: list[str],
    emit: Callable[[dict], None],
    on_scenario: Callable[[str], None] | None,
) -> None:
    """Prefix caching requires two engine instances (cache off / cache on)."""
    if config.backend.name == "naive":
        log.info("Skipping prefix-cache comparison: unsupported by the HF backend.")
        return
    if on_scenario:
        on_scenario("prefix_cache")

    for enabled in (False, True):
        variant = _with_cache_flag(config, enabled)
        backend = create_backend(variant)
        try:
            backend.load()
            _warmup(backend, variant)
            scenario_impl.run_prefix_cache(variant, backend, emit, cache_enabled=enabled, rounds=3)
        except Exception as exc:  # noqa: BLE001 - one variant failing must not kill the run
            log.error("Prefix-cache variant (enabled=%s) failed: %s", enabled, exc)
            emit(
                {
                    "scenario": "prefix_cache",
                    "level": "aggregate",
                    "backend": variant.backend.name,
                    "serving_mode": backend.serving_mode,
                    "group": "cache_on" if enabled else "cache_off",
                    "success": False,
                    "error": str(exc),
                }
            )
        finally:
            with contextlib.suppress(Exception):
                backend.close()
            release_gpu(wait_s=1.0)
