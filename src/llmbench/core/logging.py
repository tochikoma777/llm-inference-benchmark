"""Logging setup shared by the CLI, runners and dashboard."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

__all__ = ["setup_logging", "get_logger"]

_CONFIGURED = False
_NOISY_LOGGERS = (
    "matplotlib",
    "PIL",
    "urllib3",
    "asyncio",
    "vllm",
    "transformers",
    "httpx",
    "filelock",
)


def setup_logging(
    level: str | int = "INFO",
    log_file: os.PathLike[str] | str | None = None,
    force: bool = False,
) -> logging.Logger:
    """Configure the root ``llmbench`` logger.

    Uses `rich` for human-friendly console output when available and falls
    back to plain :class:`logging.StreamHandler` otherwise.
    """
    global _CONFIGURED
    root = logging.getLogger("llmbench")
    if _CONFIGURED and not force:
        return root

    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    root.setLevel(level)
    root.handlers.clear()
    root.propagate = False

    fmt = "%(message)s"
    try:  # pragma: no cover - depends on optional dependency
        from rich.console import Console
        from rich.logging import RichHandler

        handler: logging.Handler = RichHandler(
            console=Console(stderr=False, soft_wrap=False),
            show_time=True,
            show_path=False,
            markup=False,
            rich_tracebacks=True,
        )
        fmt = "%(message)s"
    except Exception:  # pragma: no cover
        handler = logging.StreamHandler(stream=sys.stdout)
        fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"

    handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
    root.addHandler(handler)

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
        )
        root.addHandler(file_handler)

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    _CONFIGURED = True
    return root


def get_logger(name: str = "llmbench") -> logging.Logger:
    """Return a namespaced child of the ``llmbench`` logger."""
    if not name.startswith("llmbench"):
        name = f"llmbench.{name}"
    return logging.getLogger(name)
