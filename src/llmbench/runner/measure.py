"""Measurement context manager.

Wraps a block of work with GPU telemetry and merges the result into the
records produced inside it. When the backend reports its own resource usage
(the mock simulator does) that takes precedence, so simulated runs still show
plausible memory/utilisation curves.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from ..backends.base import Backend
from ..core.config import MonitorConfig
from ..core.logging import get_logger
from ..metrics.gpu_monitor import GpuMonitor

__all__ = ["Telemetry", "Measurement"]

log = get_logger("runner.measure")


@dataclass
class Telemetry:
    peak_memory_mb: float | None = None
    baseline_memory_mb: float | None = None
    kv_cache_delta_mb: float | None = None
    avg_gpu_util_pct: float | None = None
    max_gpu_util_pct: float | None = None
    energy_j: float | None = None
    samples: int = 0
    source: str = "none"

    def as_fields(self) -> dict[str, Any]:
        return {
            "peak_memory_mb": self.peak_memory_mb,
            "avg_gpu_util_pct": self.avg_gpu_util_pct,
            "max_gpu_util_pct": self.max_gpu_util_pct,
            "kv_cache_delta_mb": self.kv_cache_delta_mb,
            "energy_j": self.energy_j,
        }


class Measurement:
    """``with Measurement(config.monitor, backend) as m: ... m.telemetry``"""

    def __init__(self, monitor_config: MonitorConfig, backend: Backend | None = None):
        self.monitor_config = monitor_config
        self.backend = backend
        self.monitor: GpuMonitor | None = None
        self.telemetry = Telemetry()

    def __enter__(self) -> Measurement:
        if self.backend is not None:
            self.backend.reset_telemetry()
        if self.monitor_config.enabled:
            self.monitor = GpuMonitor(
                interval_ms=self.monitor_config.interval_ms,
                device_index=self.monitor_config.device_index,
                enabled=True,
            ).start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        gpu: dict[str, Any] = {}
        if self.monitor is not None:
            self.monitor.stop()
            gpu = self.monitor.summary()
        backend_telemetry = self.backend.telemetry() if self.backend is not None else {}

        source = "gpu"
        if backend_telemetry.get("peak_memory_mb") is not None:
            # Simulated backend: prefer its model over an idle GPU reading.
            gpu = {**gpu, **{k: v for k, v in backend_telemetry.items() if v is not None}}
            source = "backend"

        self.telemetry = Telemetry(
            peak_memory_mb=gpu.get("peak_memory_mb"),
            baseline_memory_mb=gpu.get("baseline_memory_mb"),
            kv_cache_delta_mb=gpu.get("kv_cache_delta_mb"),
            avg_gpu_util_pct=gpu.get("avg_gpu_util_pct"),
            max_gpu_util_pct=gpu.get("max_gpu_util_pct"),
            energy_j=gpu.get("energy_j"),
            samples=int(gpu.get("samples") or 0),
            source=source if gpu else "none",
        )

    def attach(self, records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        """Copy telemetry fields onto records (in place) and return them."""
        fields = self.telemetry.as_fields()
        out = []
        for record in records:
            for key, value in fields.items():
                if record.get(key) is None and value is not None:
                    record[key] = value
            out.append(record)
        return out
