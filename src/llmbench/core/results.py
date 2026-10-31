"""Result persistence and loading.

Each benchmark run is stored in its own directory::

    data/results/<run_id>/
        metadata.json     # environment, config snapshot, run status
        results.jsonl     # one JSON object per measurement
        summary.json      # aggregated statistics per (scenario, backend, …)

The on-disk format is append-friendly (JSONL) and human-inspectable, which
matters for a project whose main value is *auditable* measurement.
"""

from __future__ import annotations

import dataclasses
import json
import os
import platform
import re
import socket
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .. import __version__
from .logging import get_logger
from .paths import results_dir

__all__ = [
    "SCHEMA_VERSION",
    "RECORD_DEFAULTS",
    "RunMetadata",
    "RunInfo",
    "RunWriter",
    "ResultStore",
    "new_run_id",
    "normalize_record",
    "aggregate_records",
]

log = get_logger("core.results")

SCHEMA_VERSION = 2

#: Canonical columns. Records may carry additional keys, but these are always
#: present so that the dashboard can rely on them.
RECORD_DEFAULTS: dict[str, Any] = {
    "run_id": None,
    "record_id": None,
    "scenario": None,  # single | concurrency | memory | prefix_cache | scheduler
    "level": "request",  # request | aggregate
    "backend": None,  # vllm | naive | mock
    "serving_mode": None,  # continuous_batch | static_batch | per_request …
    "model_id": None,
    "group": None,  # display label (e.g. "cache_on")
    "bucket": None,  # aggregation key — records sharing a bucket are averaged
    "repeat": 0,
    "request_index": None,
    "concurrency": None,
    "prompt_len": None,
    "output_len": None,
    "input_tokens": None,
    "output_tokens": None,
    "arrival_ms": None,
    "start_ms": None,
    "queue_ms": None,
    "ttft_ms": None,
    "tpot_ms": None,
    "e2e_ms": None,
    "finish_ms": None,
    "total_time_ms": None,
    "throughput_tok_s": None,
    "peak_memory_mb": None,
    "avg_gpu_util_pct": None,
    "max_gpu_util_pct": None,
    "kv_cache_delta_mb": None,
    "success": True,
    "error": None,
    "timestamp": None,
}


def _slug(text: str, max_len: int = 40) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", text.strip().lower()).strip("-")
    return (slug or "run")[:max_len]


def new_run_id(name: str = "run") -> str:
    """Create a sortable, human-readable run identifier."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    suffix = uuid.uuid4().hex[:6]
    return f"{stamp}-{_slug(name)}-{suffix}"


def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    """Return ``record`` with canonical keys present and JSON-safe values."""
    out: dict[str, Any] = dict(RECORD_DEFAULTS)
    out.update({k: v for k, v in record.items() if v is not None or k not in out})
    for key, value in list(out.items()):
        if isinstance(value, (datetime,)):
            out[key] = value.isoformat()
        elif dataclasses.is_dataclass(value):
            out[key] = dataclasses.asdict(value)  # type: ignore[arg-type]
    return out


@dataclass
class RunMetadata:
    """Everything needed to interpret a run without re-reading the config."""

    run_id: str
    name: str = "run"
    description: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    finished_at: str | None = None
    duration_s: float | None = None
    status: str = "running"  # running | completed | failed
    backend: str | None = None
    serving_mode: str | None = None
    model_id: str | None = None
    model_path: str | None = None
    tags: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    system: dict[str, Any] = field(default_factory=dict)
    scenarios: list[str] = field(default_factory=list)
    notes: str = ""
    error: str | None = None
    schema_version: int = SCHEMA_VERSION
    package_version: str = __version__

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunMetadata:
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self, directory: os.PathLike[str] | str) -> Path:
        path = Path(directory) / "metadata.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return path


@dataclass
class RunInfo:
    """Lightweight descriptor used to list runs without loading all records."""

    run_id: str
    path: Path
    metadata: RunMetadata
    num_records: int
    mtime: float

    @property
    def label(self) -> str:
        meta = self.metadata
        bits = [meta.backend or "?"]
        if meta.serving_mode:
            bits.append(meta.serving_mode)
        if meta.model_id:
            bits.append(Path(meta.model_id).name)
        return f"{self.run_id}  ({' / '.join(bits)})"


class RunWriter:
    """Accumulate records and persist a run directory.

    Usage::

        with RunWriter(meta) as writer:
            writer.add(record)
    """

    def __init__(self, metadata: RunMetadata, directory: Path | None = None):
        self.metadata = metadata
        self.dir = Path(directory) if directory else results_dir() / metadata.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self._records: list[dict[str, Any]] = []
        self._closed = False
        self._t0 = datetime.now(timezone.utc)

    # -- writing ---------------------------------------------------------
    def add(self, record: dict[str, Any]) -> None:
        rec = normalize_record(record)
        if rec.get("run_id") is None:
            rec["run_id"] = self.metadata.run_id
        if rec.get("record_id") is None:
            rec["record_id"] = f"r{len(self._records):06d}"
        if rec.get("model_id") is None:
            rec["model_id"] = self.metadata.model_id
        if rec.get("backend") is None:
            rec["backend"] = self.metadata.backend
        if rec.get("serving_mode") is None:
            rec["serving_mode"] = self.metadata.serving_mode
        if rec.get("timestamp") is None:
            rec["timestamp"] = datetime.now(timezone.utc).isoformat()
        self._records.append(rec)

    def add_many(self, records: Iterable[dict[str, Any]]) -> None:
        for rec in records:
            self.add(rec)

    def extend(self, records: Iterable[dict[str, Any]]) -> None:
        self.add_many(records)

    @property
    def records(self) -> list[dict[str, Any]]:
        return list(self._records)

    # -- lifecycle -------------------------------------------------------
    def write(self) -> Path:
        """Flush metadata, records and aggregates to disk."""
        meta = self.metadata
        meta.finished_at = datetime.now(timezone.utc).isoformat()
        meta.duration_s = (datetime.now(timezone.utc) - self._t0).total_seconds()
        if meta.status == "running":
            meta.status = "completed"

        with (self.dir / "results.jsonl").open("w", encoding="utf-8") as fh:
            for rec in self._records:
                fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

        summary = aggregate_records(self._records)
        (self.dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
        meta.save(self.dir)
        return self.dir

    def fail(self, error: str) -> Path:
        self.metadata.status = "failed"
        self.metadata.error = error
        return self.write()

    def __enter__(self) -> RunWriter:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc is not None:
            self.fail(f"{exc_type.__name__}: {exc}")
        elif not self._closed:
            self.write()
            self._closed = True


def aggregate_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group request-level records and compute statistics per bucket.

    Records are grouped by ``(scenario, backend, serving_mode, bucket,
    concurrency, repeat)``. Scenarios choose ``bucket`` deliberately: the
    latency sweep buckets by grid point, while a concurrency run puts every
    request in one bucket so the aggregate reflects the whole load.
    """
    from ..metrics.stats import safe_div, summarize

    buckets: dict[tuple, list[dict[str, Any]]] = {}
    # ``run_id`` is part of the key: makespan is only meaningful within a
    # single run, so two runs of the same configuration must never be merged
    # into one bucket.
    keys = ("scenario", "backend", "serving_mode", "bucket", "concurrency", "repeat", "run_id")
    for rec in records:
        if rec.get("level") == "aggregate":
            continue
        if not rec.get("success", True):
            continue
        bucket_key = tuple(rec.get(k) for k in keys)
        buckets.setdefault(bucket_key, []).append(rec)

    def _first(items: list[dict[str, Any]], key: str):
        for item in items:
            if item.get(key) is not None:
                return item[key]
        return None

    def _avg(items: list[dict[str, Any]], key: str) -> float | None:
        vals: list[float] = []
        for item in items:
            value = item.get(key)
            if value is not None:
                vals.append(float(value))
        return sum(vals) / len(vals) if vals else None

    rows: list[dict[str, Any]] = []
    for bucket_key, items in buckets.items():
        row = dict(zip(keys, bucket_key, strict=False))
        row["level"] = "aggregate"
        row["num_requests"] = len(items)
        row["group"] = _first(items, "group") or bucket_key[3]

        ttft = [i.get("ttft_ms") for i in items]
        tpot = [i.get("tpot_ms") for i in items]
        e2e = [i.get("e2e_ms") for i in items]

        ttft_stats = summarize(ttft)
        tpot_stats = summarize(tpot)
        e2e_stats = summarize(e2e)
        total_tokens = sum(int(i.get("output_tokens") or 0) for i in items)

        # Makespan: from the first scheduled arrival to the last completion.
        # This is the correct denominator for throughput under non-burst
        # arrival patterns (burst mode reduces to max(e2e)).
        arrivals: list[float] = [float(i.get("arrival_ms") or 0.0) for i in items]
        finishes: list[float] = []
        for item in items:
            finish = item.get("finish_ms")
            if finish is None:
                finish = float(item.get("arrival_ms") or 0.0) + float(item.get("e2e_ms") or 0.0)
            finishes.append(float(finish))
        makespan_ms = (max(finishes) - min(arrivals)) if finishes else 0.0
        wall_s = safe_div(makespan_ms, 1000.0, 0.0)

        row.update(
            {
                "prompt_len": round(_avg(items, "prompt_len") or 0.0, 1),
                "output_len": round(_avg(items, "output_len") or 0.0, 1),
                "target_prompt_len": _first(items, "target_prompt_len"),
                "requested_output_len": _first(items, "requested_output_len"),
                "ttft_ms": ttft_stats["mean"],
                "ttft_ms_mean": ttft_stats["mean"],
                "ttft_ms_p50": ttft_stats["p50"],
                "ttft_ms_p95": ttft_stats["p95"],
                "ttft_ms_p99": ttft_stats["p99"],
                "tpot_ms": tpot_stats["mean"],
                "tpot_ms_mean": tpot_stats["mean"],
                "e2e_ms": e2e_stats["mean"],
                "e2e_ms_mean": e2e_stats["mean"],
                "e2e_ms_p95": e2e_stats["p95"],
                "e2e_ms_p99": e2e_stats["p99"],
                "queue_ms": _avg(items, "queue_ms"),
                "output_tokens": total_tokens,
                "makespan_ms": round(makespan_ms, 3),
                "throughput_tok_s": safe_div(total_tokens, wall_s, 0.0),
                "peak_memory_mb": max((i.get("peak_memory_mb") or 0) for i in items),
                "avg_gpu_util_pct": safe_div(
                    sum(i.get("avg_gpu_util_pct") or 0 for i in items), len(items), 0.0
                ),
            }
        )
        rows.append(row)
    rows.sort(key=lambda r: tuple(str(r.get(k)) for k in keys))
    return rows


class ResultStore:
    """Read access to a directory of runs."""

    def __init__(self, root: os.PathLike[str] | str | None = None):
        self.root = Path(root) if root else results_dir()

    # -- discovery -------------------------------------------------------
    def run_dirs(self) -> list[Path]:
        """Directories containing a ``metadata.json``.

        Both ``<root>/<run>/`` and one level of grouping
        (``<root>/<group>/<run>/``, e.g. the demo suite) are discovered, so a
        grouped run is never invisible to ``llmbench runs`` or the dashboard.
        """
        if not self.root.exists():
            return []
        dirs = {
            marker.parent
            for pattern in ("*/metadata.json", "*/*/metadata.json")
            for marker in self.root.glob(pattern)
            if marker.is_file()
        }
        return sorted(dirs, key=lambda p: p.stat().st_mtime, reverse=True)

    def runs(self) -> list[RunInfo]:
        infos: list[RunInfo] = []
        for directory in self.run_dirs():
            try:
                meta = RunMetadata.from_dict(
                    json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
                )
            except Exception as exc:  # pragma: no cover - corrupted run
                log.warning("Skipping unreadable run %s: %s", directory.name, exc)
                continue
            count = 0
            results_file = directory / "results.jsonl"
            if results_file.exists():
                with results_file.open("r", encoding="utf-8") as fh:
                    count = sum(1 for line in fh if line.strip())
            infos.append(
                RunInfo(
                    run_id=meta.run_id or directory.name,
                    path=directory,
                    metadata=meta,
                    num_records=count,
                    mtime=directory.stat().st_mtime,
                )
            )
        return infos

    def latest(self, n: int = 1) -> list[RunInfo]:
        return self.runs()[:n]

    def get(self, run_id: str) -> RunInfo | None:
        for info in self.runs():
            if info.run_id == run_id:
                return info
        return None

    # -- loading ---------------------------------------------------------
    def load_records(self, run_id: str) -> list[dict[str, Any]]:
        info = self.get(run_id)
        if info is None:
            raise KeyError(f"run not found: {run_id}")
        return self._read_jsonl(info.path / "results.jsonl")

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    log.warning("Skipping malformed JSONL line in %s", path)
        return records

    def load_all(self, run_ids: Iterable[str] | None = None):
        """Return ``(DataFrame, [RunInfo, ...])`` for the selected runs.

        The DataFrame has one row per measurement plus run metadata columns
        (``run_name``, ``run_tags``, ``run_created_at``) for faceting.
        """
        import pandas as pd

        wanted = set(run_ids) if run_ids else None
        infos = [i for i in self.runs() if wanted is None or i.run_id in wanted]
        rows: list[dict[str, Any]] = []
        for info in infos:
            meta = info.metadata
            for rec in self._read_jsonl(info.path / "results.jsonl"):
                rec = normalize_record(rec)
                rec["run_id"] = info.run_id
                rec["run_name"] = meta.name
                rec["run_created_at"] = meta.created_at
                rec["run_tags"] = ",".join(meta.tags)
                rec["is_mock"] = bool(meta.system.get("mock", False))
                rows.append(rec)

        if not rows:
            return pd.DataFrame(columns=list(RECORD_DEFAULTS) + ["run_name"]), infos
        df = pd.DataFrame(rows)
        return df, infos

    # -- convenience -----------------------------------------------------
    def export_csv(
        self, path: os.PathLike[str] | str, run_ids: Iterable[str] | None = None
    ) -> Path:
        df, _ = self.load_all(run_ids)
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        return out

    def summary(self, run_id: str | None = None) -> list[dict[str, Any]]:
        if run_id:
            info = self.get(run_id)
        else:
            latest = self.latest(1)
            info = latest[0] if latest else None
        if info is None:
            return []
        path = info.path / "summary.json"
        if not path.exists():
            return aggregate_records(self._read_jsonl(info.path / "results.jsonl"))
        return json.loads(path.read_text(encoding="utf-8"))

    def delete(self, run_id: str, dry_run: bool = False) -> None:
        info = self.get(run_id)
        if info is None:
            raise KeyError(f"run not found: {run_id}")
        import shutil

        if dry_run:
            log.info("[dry-run] would delete %s", info.path)
            return
        shutil.rmtree(info.path)
        log.info("Deleted run %s", run_id)


def build_metadata(
    config: Any,
    scenarios: list[str] | None = None,
    system: dict[str, Any] | None = None,
) -> RunMetadata:
    """Create :class:`RunMetadata` from a :class:`~llmbench.core.config.RunConfig`."""
    model_path = config.model.path
    return RunMetadata(
        run_id=new_run_id(config.name),
        name=config.name,
        description=config.description,
        backend=config.backend.name,
        serving_mode=config.backend.effective_serving_mode,
        model_id=Path(model_path).name if model_path else None,
        model_path=str(model_path) if model_path else None,
        tags=list(config.tags),
        config=config.to_dict(),
        system=system or {},
        scenarios=list(scenarios or []),
    )


def default_system_info() -> dict[str, Any]:
    """Minimal, always-available environment fingerprint."""
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "machine": platform.machine(),
    }
