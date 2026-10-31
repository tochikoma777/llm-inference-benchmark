"""Structured configuration for benchmark runs.

Configuration is a small, explicit dataclass tree that can be loaded from
YAML/JSON, overridden by environment variables and by ``key=value`` CLI
overrides. Keeping it a dataclass (rather than a nested dict) means typos in
a config file fail loudly instead of silently producing ``None`` deep inside
a benchmark.
"""

from __future__ import annotations

import dataclasses
import json
import os
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

import yaml

from .paths import project_root

__all__ = [
    "ConfigError",
    "ValidationError",
    "ModelConfig",
    "BackendConfig",
    "ArrivalConfig",
    "WorkloadConfig",
    "MonitorConfig",
    "OutputConfig",
    "MockConfig",
    "ScenarioConfig",
    "RunConfig",
    "load_config",
]


class ConfigError(ValueError):
    """Raised when a configuration file cannot be parsed or validated."""


class ValidationError(ConfigError):
    """Raised when configuration values are semantically invalid."""


SERVING_MODES = {
    "auto",
    "continuous_batch",
    "static_batch",
    "per_request",
    "offline",
    "async",
}
BACKEND_NAMES = {"vllm", "naive", "mock"}
ARRIVAL_MODES = {"burst", "constant", "poisson"}

#: Default serving mode when ``backend.serving_mode == "auto"``.
_DEFAULT_SERVING_MODE = {
    "vllm": "continuous_batch",
    "naive": "static_batch",
    "mock": "continuous_batch",
}


@dataclass
class ModelConfig:
    """Model / tokenizer description."""

    path: str | None = None
    tokenizer_path: str | None = None
    dtype: str = "float16"
    max_model_len: int = 8192
    trust_remote_code: bool = True
    quantization: str | None = None
    revision: str | None = None
    device: str = "cuda"
    extra: dict[str, Any] = field(default_factory=dict)

    def resolve_path(self) -> str:
        """Resolve the model path, honouring ``LLMBENCH_MODEL_PATH``."""
        candidate = self.path or os.environ.get("LLMBENCH_MODEL_PATH")
        if not candidate:
            raise ConfigError(
                "No model path configured. Set `model.path` in your config, pass "
                "`--model-path`, or export LLMBENCH_MODEL_PATH."
            )
        resolved = Path(candidate).expanduser()
        if not resolved.exists():
            raise ConfigError(
                f"Model path does not exist: {resolved}\n"
                "Tip: `llmbench fetch-model --model qwen/Qwen2.5-7B-Instruct` "
                "downloads it from ModelScope."
            )
        return str(resolved)

    def resolve_tokenizer_path(self) -> str:
        if self.tokenizer_path:
            return str(Path(self.tokenizer_path).expanduser())
        return self.resolve_path()


@dataclass
class BackendConfig:
    """Inference backend selection and engine tuning knobs."""

    name: str = "mock"
    serving_mode: str = "auto"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.85
    enable_prefix_caching: bool = False
    max_num_seqs: int = 256
    max_batch_size: int = 16
    enforce_eager: bool = False
    device: str = "cuda"
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def effective_serving_mode(self) -> str:
        if self.serving_mode == "auto":
            return _DEFAULT_SERVING_MODE.get(self.name, "per_request")
        return self.serving_mode

    def validate(self) -> None:
        if self.name not in BACKEND_NAMES:
            raise ValidationError(
                f"backend.name={self.name!r} is not one of {sorted(BACKEND_NAMES)}"
            )
        if self.serving_mode not in SERVING_MODES:
            raise ValidationError(
                f"backend.serving_mode={self.serving_mode!r} is not one of "
                f"{sorted(SERVING_MODES)}"
            )
        if not 0.0 < self.gpu_memory_utilization <= 1.0:
            raise ValidationError("backend.gpu_memory_utilization must be in (0, 1]")
        if self.tensor_parallel_size < 1:
            raise ValidationError("backend.tensor_parallel_size must be >= 1")
        if self.max_batch_size < 1:
            raise ValidationError("backend.max_batch_size must be >= 1")


@dataclass
class ArrivalConfig:
    """Request arrival process used by the load generator."""

    mode: str = "burst"
    rate_rps: float = 8.0
    jitter: float = 0.0

    def validate(self) -> None:
        if self.mode not in ARRIVAL_MODES:
            raise ValidationError(
                f"workload.arrival.mode={self.mode!r} is not one of {sorted(ARRIVAL_MODES)}"
            )
        if self.rate_rps <= 0:
            raise ValidationError("workload.arrival.rate_rps must be > 0")
        if self.jitter < 0:
            raise ValidationError("workload.arrival.jitter must be >= 0")


@dataclass
class WorkloadConfig:
    """What to send to the model, and how."""

    dataset: str | None = None
    num_requests: int = 32
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    warmup_runs: int = 2
    request_timeout_s: float = 600.0
    arrival: ArrivalConfig = field(default_factory=ArrivalConfig)

    def validate(self) -> None:
        self.arrival.validate()
        if self.num_requests < 1:
            raise ValidationError("workload.num_requests must be >= 1")
        if self.max_new_tokens < 1:
            raise ValidationError("workload.max_new_tokens must be >= 1")
        if self.warmup_runs < 0:
            raise ValidationError("workload.warmup_runs must be >= 0")
        if not 0.0 <= self.temperature <= 5.0:
            raise ValidationError("workload.temperature must be in [0, 5]")


@dataclass
class MonitorConfig:
    """GPU telemetry sampling."""

    enabled: bool = True
    interval_ms: int = 100
    device_index: int = 0

    def validate(self) -> None:
        if self.interval_ms < 10:
            raise ValidationError("monitor.interval_ms must be >= 10")
        if self.device_index < 0:
            raise ValidationError("monitor.device_index must be >= 0")


@dataclass
class OutputConfig:
    """Where results are written."""

    dir: str | None = None
    tag: str | None = None
    save_text: bool = False  # persist generated text (privacy/size trade-off)


@dataclass
class MockConfig:
    """Deterministic simulation used to exercise the pipeline without a GPU."""

    time_scale: float = 0.05  # 0.05 => simulated 1s costs 50ms of wall clock
    gpu_profile: str = "rtx3090"
    noise: float = 0.03
    seed: int = 7

    def validate(self) -> None:
        if self.time_scale <= 0:
            raise ValidationError("mock.time_scale must be > 0")
        if not 0.0 <= self.noise < 1.0:
            raise ValidationError("mock.noise must be in [0, 1)")


@dataclass
class SimConfig:
    """Parameters of the scheduler simulator (pure software, no GPU)."""

    num_requests: int = 120
    short_ratio: float = 0.7
    arrival_mean_ms: float = 1000.0
    seed: int = 42
    policies: list[str] = field(default_factory=lambda: ["fcfs", "sjf", "priority", "sjf_aging"])
    max_num_seqs: int = 12
    block_size: int = 16
    kv_pool_mb: float = 20000.0
    kv_mb_per_token: float = 0.5
    prefill_tok_per_ms: float = 2.5
    decode_step_ms: float = 20.0
    decode_batch_factor: float = 0.12
    sjf_aging_tau_ms: float = 2000.0

    def validate(self) -> None:
        if self.num_requests < 1:
            raise ValidationError("sim.num_requests must be >= 1")
        if not 0.0 <= self.short_ratio <= 1.0:
            raise ValidationError("sim.short_ratio must be in [0, 1]")
        if self.max_num_seqs < 1:
            raise ValidationError("sim.max_num_seqs must be >= 1")
        if self.block_size < 1:
            raise ValidationError("sim.block_size must be >= 1")
        if self.prefill_tok_per_ms <= 0 or self.decode_step_ms <= 0:
            raise ValidationError("sim throughput parameters must be > 0")
        if self.decode_batch_factor < 0:
            raise ValidationError("sim.decode_batch_factor must be >= 0")
        known = {"fcfs", "sjf", "priority", "sjf_aging", "round_robin"}
        unknown = [p for p in self.policies if p not in known]
        if unknown:
            raise ValidationError(
                f"sim.policies contains unknown entries {unknown}; known: {sorted(known)}"
            )


@dataclass
class ScenarioConfig:
    """Which benchmarks to execute, and at which grid points."""

    single: bool = True
    concurrency: bool = True
    memory: bool = True
    prefix_cache: bool = True
    scheduler: bool = True
    concurrency_levels: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    prompt_lengths: list[int] = field(default_factory=lambda: [256, 512, 1024, 2048])
    output_lengths: list[int] = field(default_factory=lambda: [1, 128, 256])
    repeats: int = 1

    def validate(self) -> None:
        if not self.concurrency_levels:
            raise ValidationError("scenarios.concurrency_levels must not be empty")
        if any(c < 1 for c in self.concurrency_levels):
            raise ValidationError("scenarios.concurrency_levels entries must be >= 1")
        if not self.prompt_lengths:
            raise ValidationError("scenarios.prompt_lengths must not be empty")
        if not self.output_lengths:
            raise ValidationError("scenarios.output_lengths must not be empty")
        if self.repeats < 1:
            raise ValidationError("scenarios.repeats must be >= 1")


@dataclass
class RunConfig:
    """Top-level configuration for a single benchmark run."""

    name: str = "default"
    description: str = ""
    seed: int = 42
    tags: list[str] = field(default_factory=list)
    model: ModelConfig = field(default_factory=ModelConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    workload: WorkloadConfig = field(default_factory=WorkloadConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    mock: MockConfig = field(default_factory=MockConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    scenarios: ScenarioConfig = field(default_factory=ScenarioConfig)
    extra: dict[str, Any] = field(default_factory=dict)

    # -- validation ------------------------------------------------------
    def validate(self) -> RunConfig:
        if not self.name.strip():
            raise ValidationError("name must not be empty")
        self.backend.validate()
        self.workload.validate()
        self.monitor.validate()
        self.mock.validate()
        self.sim.validate()
        self.scenarios.validate()
        return self

    # -- (de)serialisation ----------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: os.PathLike[str] | str) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        return out


# ---------------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------------


def _unwrap_optional(tp: Any) -> Any:
    """Return ``T`` for ``Optional[T]`` / ``T | None``, else ``tp``."""
    origin = get_origin(tp)
    # ``typing.Union`` for Optional[...], ``types.UnionType`` for X | None.
    union_types: tuple[Any, ...] = (Union, getattr(types, "UnionType", Union))
    if origin in union_types:
        args = [a for a in get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return tp


def _construct(cls: Any, data: dict[str, Any], path: str = "") -> Any:
    """Recursively build a dataclass from a plain dict."""
    if not isinstance(data, dict):
        raise ConfigError(f"expected a mapping at {path or '<root>'}, got {type(data).__name__}")

    hints = get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    kwargs: dict[str, Any] = {}
    extra: dict[str, Any] = dict(data.get("extra") or {})

    for key, value in data.items():
        where = f"{path}.{key}" if path else key
        if key not in known:
            extra[key] = value
            continue
        target = _unwrap_optional(hints.get(key, Any))
        if dataclasses.is_dataclass(target) and isinstance(value, dict):
            kwargs[key] = _construct(target, value, where)
        elif dataclasses.is_dataclass(target) and isinstance(value, target):  # type: ignore[arg-type]
            kwargs[key] = value
        else:
            kwargs[key] = value

    if "extra" in known:
        kwargs["extra"] = extra
    try:
        return cls(**kwargs)
    except TypeError as exc:  # pragma: no cover - defensive
        raise ConfigError(f"invalid configuration at {path or '<root>'}: {exc}") from exc


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _coerce_scalar(raw: str) -> Any:
    """Parse ``"8"`` -> 8, ``"true"`` -> True, ``"[1,2]"`` -> [1, 2]."""
    text = raw.strip()
    if text == "":
        return ""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def apply_overrides(data: dict[str, Any], overrides: dict[str, Any] | list[str]) -> dict[str, Any]:
    """Apply dotted ``key=value`` overrides onto a nested dict."""
    items = (
        overrides.items()
        if isinstance(overrides, dict)
        else (tuple(o.split("=", 1)) for o in overrides if "=" in o)
    )
    out = dict(data)
    for dotted, raw in items:
        value = _coerce_scalar(raw) if isinstance(raw, str) else raw
        cursor = out
        parts = dotted.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
            if not isinstance(cursor, dict):
                raise ConfigError(f"cannot set {dotted!r}: {part!r} is not a mapping")
        cursor[parts[-1]] = value
    return out


def _env_overrides() -> dict[str, Any]:
    """Small set of environment overrides for CI convenience."""
    mapping = {
        "LLMBENCH_MODEL_PATH": "model.path",
        "LLMBENCH_BACKEND": "backend.name",
        "LLMBENCH_SERVING_MODE": "backend.serving_mode",
        "LLMBENCH_SEED": "seed",
        "LLMBENCH_RESULTS_DIR": "output.dir",
    }
    return {target: os.environ[var] for var, target in mapping.items() if os.environ.get(var)}


DEFAULT_CONFIG_PATH = project_root() / "config" / "default.yaml"


def load_config(
    path: os.PathLike[str] | str | None = None,
    overrides: dict[str, Any] | list[str] | None = None,
    use_env: bool = True,
) -> RunConfig:
    """Load a :class:`RunConfig` from YAML/JSON with optional overrides.

    Parameters
    ----------
    path:
        Config file to load. ``None`` means "use built-in defaults"
        (``config/default.yaml`` if present, otherwise dataclass defaults).
    overrides:
        ``key=value`` strings or a nested dict applied on top of the file.
    use_env:
        Apply ``LLMBENCH_*`` environment overrides.
    """
    data: dict[str, Any] = {}

    if path is not None:
        cfg_path = Path(path).expanduser()
        if not cfg_path.exists():
            raise ConfigError(f"Configuration file not found: {cfg_path}")
        text = cfg_path.read_text(encoding="utf-8")
        try:
            loaded = (
                yaml.safe_load(text)
                if cfg_path.suffix.lower() in {".yaml", ".yml"}
                else json.loads(text)
            )
        except (yaml.YAMLError, json.JSONDecodeError) as exc:
            raise ConfigError(f"Cannot parse {cfg_path}: {exc}") from exc
        if loaded is not None and not isinstance(loaded, dict):
            raise ConfigError(f"{cfg_path} must contain a mapping at the top level")
        data = loaded or {}
    elif DEFAULT_CONFIG_PATH.exists():
        data = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")) or {}

    if use_env:
        data = apply_overrides(data, _env_overrides())
    if overrides:
        data = apply_overrides(data, overrides)

    config = _construct(RunConfig, data)
    return config.validate()
