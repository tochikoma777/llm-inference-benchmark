"""GPU telemetry sampling.

Two backends are supported:

``nvml``
    Preferred. Sub-millisecond reads, no subprocess spawn, and exposes power
    draw and temperature. Requires ``pynvml`` / ``nvidia-ml-py``.
``nvidia-smi``
    Fallback for environments where NVML is unavailable. Spawns a process per
    sample (~30–60 ms), so it silently clamps to a coarser interval.

Measurement notes
-----------------
* ``baseline_memory_mb`` is captured right after :meth:`GpuMonitor.start`, so
  ``kv_cache_delta_mb`` isolates memory *added by generation* (weights already
  resident) — this is what the memory-scaling scenario reports.
* Sampling is done in a daemon thread so it never blocks the benchmark loop.
"""

from __future__ import annotations

import contextlib
import shutil
import subprocess
import threading
import time
import warnings
from dataclasses import asdict, dataclass
from typing import Any

from ..core.logging import get_logger

__all__ = ["GpuSample", "GpuMonitor", "gpu_available", "gpu_name"]

log = get_logger("metrics.gpu")


@dataclass
class GpuSample:
    timestamp_s: float
    memory_used_mb: float
    memory_total_mb: float
    gpu_util_pct: float
    temperature_c: float | None = None
    power_w: float | None = None

    def as_dict(self) -> dict[str, float | None]:
        return asdict(self)


def _nvml_available() -> bool:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
        return True
    except Exception:
        return False


def _smi_available() -> bool:
    return shutil.which("nvidia-smi") is not None


def gpu_available() -> bool:
    """True when any GPU telemetry source can be opened."""
    return _nvml_available() or _smi_available()


def gpu_name(device_index: int = 0) -> str | None:
    """Best-effort human-readable GPU name."""
    if _nvml_available():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                import pynvml  # noqa: PLC0415

            pynvml.nvmlInit()
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
                return pynvml.nvmlDeviceGetName(handle)
            finally:
                pynvml.nvmlShutdown()
        except Exception:
            pass
    if _smi_available():
        try:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name",
                    "--format=csv,noheader",
                    "-i",
                    str(device_index),
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
            return out.stdout.strip().splitlines()[0] if out.stdout.strip() else None
        except Exception:
            return None
    return None


class _NvmlReader:
    """Thin NVML wrapper that keeps a device handle alive for the sampler."""

    def __init__(self, device_index: int = 0):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import pynvml  # noqa: PLC0415

        self._nvml = pynvml
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)

    def read(self) -> GpuSample:
        nvml = self._nvml
        mem = nvml.nvmlDeviceGetMemoryInfo(self.handle)
        util = nvml.nvmlDeviceGetUtilizationRates(self.handle)
        temp = power = None
        with contextlib.suppress(Exception):
            temp = float(nvml.nvmlDeviceGetTemperature(self.handle, nvml.NVML_TEMPERATURE_GPU))
        with contextlib.suppress(Exception):
            power = nvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
        return GpuSample(
            timestamp_s=time.perf_counter(),
            memory_used_mb=mem.used / 1024**2,
            memory_total_mb=mem.total / 1024**2,
            gpu_util_pct=float(util.gpu),
            temperature_c=temp,
            power_w=power,
        )

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._nvml.nvmlShutdown()


class _SmiReader:
    """Fallback reader built on ``nvidia-smi --query-gpu``."""

    QUERY = "memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw"

    def __init__(self, device_index: int = 0):
        self.device_index = device_index

    def read(self) -> GpuSample:
        out = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={self.QUERY}",
                "--format=csv,noheader,nounits",
                "-i",
                str(self.device_index),
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        parts = [p.strip() for p in out.stdout.strip().splitlines()[0].split(",")]

        def _num(i: int) -> float | None:
            try:
                return float(parts[i])
            except (IndexError, ValueError):
                return None

        return GpuSample(
            timestamp_s=time.perf_counter(),
            memory_used_mb=_num(0) or 0.0,
            memory_total_mb=_num(1) or 0.0,
            gpu_util_pct=_num(2) or 0.0,
            temperature_c=_num(3),
            power_w=_num(4),
        )

    def close(self) -> None:
        return None


class GpuMonitor:
    """Background sampler for memory / utilisation / power."""

    def __init__(self, interval_ms: int = 100, device_index: int = 0, enabled: bool = True):
        self.interval_s = max(interval_ms, 10) / 1000.0
        self.device_index = device_index
        self.enabled = enabled and gpu_available()
        self.source: str | None = None
        self.samples: list[GpuSample] = []
        self.baseline_memory_mb: float | None = None
        self.error: str | None = None

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._device_name: str | None = None

    # -- lifecycle -------------------------------------------------------
    def start(self) -> GpuMonitor:
        if not self.enabled:
            return self
        self.samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="gpu-monitor", daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        reader: Any = None
        try:
            if _nvml_available():
                try:
                    reader = _NvmlReader(self.device_index)
                    self.source = "nvml"
                    self._device_name = gpu_name(self.device_index)
                except Exception as exc:
                    log.debug("NVML unavailable (%s), falling back to nvidia-smi", exc)
            if reader is None and _smi_available():
                reader = _SmiReader(self.device_index)
                self.source = "nvidia-smi"
                # nvidia-smi costs ~40 ms per call; avoid pathological rates.
                self.interval_s = max(self.interval_s, 0.2)
            if reader is None:
                self.enabled = False
                self.error = "no GPU telemetry source available"
                return

            while not self._stop.is_set():
                try:
                    sample = reader.read()
                except Exception as exc:  # pragma: no cover - driver hiccup
                    self.error = str(exc)
                    time.sleep(self.interval_s)
                    continue
                if self.baseline_memory_mb is None:
                    self.baseline_memory_mb = sample.memory_used_mb
                self.samples.append(sample)
                self._stop.wait(self.interval_s)
        finally:
            if reader is not None:
                reader.close()

    def stop(self) -> GpuMonitor:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=3.0)
            self._thread = None
        return self

    def __enter__(self) -> GpuMonitor:
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    # -- derived metrics -------------------------------------------------
    @property
    def device_name(self) -> str | None:
        return self._device_name

    def peak_memory_mb(self) -> float | None:
        return max((s.memory_used_mb for s in self.samples), default=None)

    def avg_gpu_util_pct(self) -> float | None:
        if not self.samples:
            return None
        return sum(s.gpu_util_pct for s in self.samples) / len(self.samples)

    def max_gpu_util_pct(self) -> float | None:
        return max((s.gpu_util_pct for s in self.samples), default=None)

    def avg_power_w(self) -> float | None:
        powers = [s.power_w for s in self.samples if s.power_w is not None]
        return sum(powers) / len(powers) if powers else None

    def kv_cache_delta_mb(self) -> float | None:
        peak = self.peak_memory_mb()
        if peak is None or self.baseline_memory_mb is None:
            return None
        return max(peak - self.baseline_memory_mb, 0.0)

    def energy_joules(self) -> float | None:
        """Trapezoidal integration of power over sampled time."""
        pts = [(s.timestamp_s, s.power_w) for s in self.samples if s.power_w is not None]
        if len(pts) < 2:
            return None
        energy = 0.0
        for (t0, p0), (t1, p1) in zip(pts, pts[1:], strict=False):
            energy += 0.5 * (p0 + p1) * max(t1 - t0, 0.0)
        return energy

    def summary(self) -> dict[str, Any]:
        avg_util = self.avg_gpu_util_pct()
        energy = self.energy_joules()
        return {
            "samples": len(self.samples),
            "source": self.source,
            "device_name": self._device_name,
            "peak_memory_mb": self.peak_memory_mb(),
            "baseline_memory_mb": self.baseline_memory_mb,
            "kv_cache_delta_mb": self.kv_cache_delta_mb(),
            "avg_gpu_util_pct": avg_util,
            "max_gpu_util_pct": self.max_gpu_util_pct(),
            "avg_power_w": self.avg_power_w(),
            "energy_j": energy,
            "error": self.error,
        }

    def series(self) -> list[dict[str, float | None]]:
        """Sampled series relative to the first sample (for timeline charts)."""
        if not self.samples:
            return []
        t0 = self.samples[0].timestamp_s
        out = []
        for s in self.samples:
            d = s.as_dict()
            d["t_rel_s"] = s.timestamp_s - t0
            d.pop("timestamp_s", None)
            out.append(d)
        return out
