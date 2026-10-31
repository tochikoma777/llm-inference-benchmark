"""Measurement primitives: statistics, GPU telemetry, environment probing."""

from __future__ import annotations

from .gpu_monitor import GpuMonitor, GpuSample, gpu_available, gpu_name
from .stats import (
    cdf_points,
    coefficient_of_variation,
    histogram,
    mean,
    percentile,
    safe_div,
    stdev,
    summarize,
)
from .system import collect_system_info, package_version

__all__ = [
    "GpuMonitor",
    "GpuSample",
    "gpu_available",
    "gpu_name",
    "cdf_points",
    "coefficient_of_variation",
    "histogram",
    "mean",
    "percentile",
    "safe_div",
    "stdev",
    "summarize",
    "collect_system_info",
    "package_version",
]
