"""Core infrastructure: configuration, logging, paths and result storage."""

from __future__ import annotations

from .config import (
    BackendConfig,
    ConfigError,
    ModelConfig,
    RunConfig,
    ScenarioConfig,
    SimConfig,
    ValidationError,
    WorkloadConfig,
    load_config,
)
from .logging import get_logger, setup_logging
from .paths import (
    cache_dir,
    data_dir,
    ensure_dir,
    figures_dir,
    logs_dir,
    project_root,
    prompts_dir,
    reports_dir,
    results_dir,
)
from .results import (
    ResultStore,
    RunInfo,
    RunMetadata,
    RunWriter,
    build_metadata,
    new_run_id,
)

__all__ = [
    "BackendConfig",
    "ConfigError",
    "ModelConfig",
    "RunConfig",
    "ScenarioConfig",
    "SimConfig",
    "ValidationError",
    "WorkloadConfig",
    "load_config",
    "get_logger",
    "setup_logging",
    "cache_dir",
    "data_dir",
    "ensure_dir",
    "figures_dir",
    "logs_dir",
    "project_root",
    "prompts_dir",
    "reports_dir",
    "results_dir",
    "ResultStore",
    "RunInfo",
    "RunMetadata",
    "RunWriter",
    "build_metadata",
    "new_run_id",
]
