"""Filesystem layout helpers.

All output locations are resolved through this module so that scripts,
the CLI and the dashboard never hard-code paths. Every location can be
overridden with an environment variable, which keeps the tool usable in
CI, Docker and on multi-user machines.

Environment overrides
---------------------
``LLMBENCH_ROOT``       repository root
``LLMBENCH_DATA_DIR``   ground truth data (prompts + results)
``LLMBENCH_RESULTS_DIR`` benchmark outputs
``LLMBENCH_REPORTS_DIR`` figures and generated reports
``LLMBENCH_CACHE_DIR``  model / misc cache
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "project_root",
    "data_dir",
    "prompts_dir",
    "results_dir",
    "reports_dir",
    "figures_dir",
    "logs_dir",
    "cache_dir",
    "ensure_dir",
    "display_path",
]


def project_root() -> Path:
    """Return the repository root (overridable via ``LLMBENCH_ROOT``)."""
    override = os.environ.get("LLMBENCH_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    # src/llmbench/core/paths.py -> parents[3] is the repo root
    return Path(__file__).resolve().parents[3]


def _dir(env_var: str, default: Path) -> Path:
    override = os.environ.get(env_var)
    return ensure_dir(Path(override).expanduser().resolve() if override else default)


def data_dir() -> Path:
    return _dir("LLMBENCH_DATA_DIR", project_root() / "data")


def prompts_dir() -> Path:
    return _dir("LLMBENCH_PROMPTS_DIR", data_dir() / "prompts")


def results_dir() -> Path:
    return _dir("LLMBENCH_RESULTS_DIR", data_dir() / "results")


def reports_dir() -> Path:
    return _dir("LLMBENCH_REPORTS_DIR", project_root() / "reports")


def figures_dir() -> Path:
    return _dir("LLMBENCH_FIGURES_DIR", reports_dir() / "figures")


def logs_dir() -> Path:
    return _dir("LLMBENCH_LOGS_DIR", project_root() / "logs")


def cache_dir() -> Path:
    return _dir("LLMBENCH_CACHE_DIR", project_root() / ".cache")


def ensure_dir(path: os.PathLike[str] | str) -> Path:
    """Create ``path`` (if needed) and return it as a :class:`Path`."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def display_path(path: os.PathLike[str] | str) -> str:
    """Return ``path`` relative to the project root when possible."""
    p = Path(path).resolve()
    try:
        return str(p.relative_to(project_root()))
    except ValueError:
        return str(p)
