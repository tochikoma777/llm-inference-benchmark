"""Environment fingerprint recorded with every run.

Versions are read from package metadata (``importlib.metadata``) instead of
importing torch/vllm, so this module is cheap and works without a GPU.
"""

from __future__ import annotations

import importlib.metadata as md
import os
import platform
import socket
import sys
from typing import Any

from .gpu_monitor import gpu_name

__all__ = ["collect_system_info", "package_version", "total_ram_gb"]


def package_version(name: str) -> str | None:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None
    except Exception:  # pragma: no cover
        return None


def total_ram_gb() -> float | None:
    """Read total RAM on Linux without psutil."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    kb = float(line.split()[1])
                    return round(kb / 1024**2, 1)
    except Exception:
        return None
    return None


def collect_system_info(mock: bool = False, gpu_index: int = 0) -> dict[str, Any]:
    """Collect a JSON-serialisable snapshot of the runtime environment."""
    info: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "total_ram_gb": total_ram_gb(),
        "packages": {
            name: package_version(name)
            for name in (
                "torch",
                "transformers",
                "vllm",
                "accelerate",
                "numpy",
                "pandas",
                "llmbench",
            )
        },
        "gpu_name": gpu_name(gpu_index),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "mock": mock,
    }
    try:
        import subprocess

        driver = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
                "-i",
                str(gpu_index),
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if driver.returncode == 0 and driver.stdout.strip():
            info["driver_version"] = driver.stdout.strip().splitlines()[0]
    except Exception:
        pass
    return info
