"""Analysis layer: turns stored runs into tidy frames and headline metrics.

Both the HTML report and the Streamlit dashboard consume :class:`Analysis`,
so a metric is defined exactly once.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ..core.logging import get_logger
from ..core.results import ResultStore, RunInfo, aggregate_records
from ..metrics.stats import safe_div

__all__ = ["Analysis"]

log = get_logger("reporting.analysis")

_NUMERIC_COLUMNS = (
    "concurrency",
    "prompt_len",
    "output_len",
    "input_tokens",
    "output_tokens",
    "arrival_ms",
    "start_ms",
    "queue_ms",
    "ttft_ms",
    "tpot_ms",
    "e2e_ms",
    "finish_ms",
    "makespan_ms",
    "throughput_tok_s",
    "peak_memory_mb",
    "avg_gpu_util_pct",
    "max_gpu_util_pct",
    "kv_cache_delta_mb",
    "ttft_ms_mean",
    "ttft_ms_p50",
    "ttft_ms_p95",
    "ttft_ms_p99",
    "tpot_ms_mean",
    "e2e_ms_mean",
    "e2e_ms_p95",
    "e2e_ms_p99",
    "num_requests",
    "target_prompt_len",
    "requested_output_len",
    "energy_j",
)

_CANONICAL = (
    "scenario",
    "level",
    "backend",
    "serving_mode",
    "group",
    "bucket",
    "repeat",
    "run_id",
    "run_name",
    "concurrency",
    "prompt_len",
    "output_len",
    "target_prompt_len",
    "requested_output_len",
    "num_requests",
    "input_tokens",
    "output_tokens",
    "arrival_ms",
    "start_ms",
    "queue_ms",
    "ttft_ms",
    "ttft_ms_mean",
    "ttft_ms_p50",
    "ttft_ms_p95",
    "ttft_ms_p99",
    "tpot_ms",
    "tpot_ms_mean",
    "e2e_ms",
    "e2e_ms_mean",
    "e2e_ms_p95",
    "e2e_ms_p99",
    "finish_ms",
    "makespan_ms",
    "throughput_tok_s",
    "peak_memory_mb",
    "avg_gpu_util_pct",
    "max_gpu_util_pct",
    "kv_cache_delta_mb",
    "energy_j",
    "success",
)


def _to_num(df: pd.DataFrame) -> pd.DataFrame:
    for column in _NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(f):
        return None
    return f


def _series_label(backend: Any, serving_mode: Any) -> str:
    """Human label that distinguishes two runs of the same engine.

    ``mock`` can emulate several serving strategies, and a real comparison
    often contains both ``vllm/continuous_batch`` and ``naive/static_batch``.
    Grouping by ``(backend, serving_mode)`` keeps those apart.
    """
    backend = str(backend) if backend is not None else "?"
    mode = str(serving_mode) if serving_mode is not None else ""
    # A bare "mock" label is meaningless (it emulates several strategies), so
    # the simulated backend always carries its serving mode.
    if backend != "mock" and mode in ("", "continuous_batch"):
        return backend
    return f"{backend} · {mode}" if mode else backend


@dataclass
class Analysis:
    """Tidy view over one or more benchmark runs."""

    raw: pd.DataFrame
    agg: pd.DataFrame
    runs: list[RunInfo] = field(default_factory=list)

    # -- construction ----------------------------------------------------
    @classmethod
    def load(
        cls,
        store: ResultStore | None = None,
        run_ids: Iterable[str] | None = None,
        latest_only: bool = False,
    ) -> Analysis:
        store = store or ResultStore()
        ids = list(run_ids) if run_ids else None
        if latest_only and ids is None:
            latest = store.latest(1)
            ids = [info.run_id for info in latest]
        raw, infos = store.load_all(ids)
        if raw.empty:
            return cls(raw=raw, agg=pd.DataFrame(columns=list(_CANONICAL)), runs=infos)

        raw = _to_num(raw)
        raw["success"] = raw.get("success", True)
        raw["series"] = [
            _series_label(b, m)
            for b, m in zip(raw.get("backend"), raw.get("serving_mode"), strict=False)
        ]
        derived = pd.DataFrame(aggregate_records(raw.to_dict("records")))
        explicit = raw[raw.get("level", "request") == "aggregate"].copy()

        # Scenarios that emit their own aggregate rows (memory, scheduler) win
        # over the derived ones: they carry extra fields the generic
        # aggregation cannot reconstruct.
        if not derived.empty and not explicit.empty:
            key_cols = [
                c
                for c in ("scenario", "backend", "serving_mode", "bucket", "concurrency")
                if c in derived.columns and c in explicit.columns
            ]

            def _key(values) -> tuple:
                # ``12`` and ``12.0`` must compare equal.
                out = []
                for value in values:
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        out.append(f"{float(value):g}")
                    else:
                        out.append(str(value))
                return tuple(out)

            explicit_keys = {_key(row) for row in explicit[key_cols].to_numpy()}
            mask = [_key(row) not in explicit_keys for row in derived[key_cols].to_numpy()]
            derived = derived[mask]

        agg = (
            pd.concat([derived, explicit], ignore_index=True, sort=False)
            if not derived.empty or not explicit.empty
            else pd.DataFrame()
        )
        if not agg.empty:
            agg = _to_num(agg)
            agg["level"] = "aggregate"
            agg["series"] = [
                _series_label(b, m)
                for b, m in zip(agg.get("backend"), agg.get("serving_mode"), strict=False)
            ]
            for column in ("run_id", "run_name", "is_mock"):
                if column in raw.columns and column not in agg.columns:
                    mapping = raw.dropna(subset=["run_id"]).set_index("run_id")[column].to_dict()
                    agg[column] = (
                        agg.get("run_id").map(mapping) if "run_id" in agg.columns else None
                    )
        return cls(raw=raw, agg=agg, runs=infos)

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _collapse_runs(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
        """Average repeated runs of the same configuration into one point.

        Running the same config three times gives three measurements of the
        same operating point; charts should show one dot, not three.
        """
        if frame.empty or "run_id" not in frame.columns or frame["run_id"].nunique() <= 1:
            return frame

        present_keys = [k for k in keys if k in frame.columns]
        if not present_keys:
            return frame

        numeric = [
            c
            for c in frame.columns
            if c not in present_keys and pd.api.types.is_numeric_dtype(frame[c])
        ]
        other = [c for c in frame.columns if c not in present_keys and c not in numeric]
        spec: dict[str, str] = dict.fromkeys(numeric, "mean")
        spec.update(dict.fromkeys(other, "first"))

        counts = (
            frame.groupby(present_keys, dropna=False)["run_id"].nunique().reset_index()
        ).rename(columns={"run_id": "n_runs"})
        out = frame.groupby(present_keys, dropna=False, as_index=False).agg(spec)
        return out.merge(counts, on=present_keys, how="left")

    # -- views -----------------------------------------------------------
    @property
    def empty(self) -> bool:
        return self.raw.empty

    @property
    def is_mock(self) -> bool:
        if "is_mock" in self.raw.columns:
            return bool(self.raw["is_mock"].fillna(False).astype(bool).any())
        return False

    def backends(self) -> list[str]:
        if self.agg.empty or "backend" not in self.agg.columns:
            return []
        return sorted(set(self.agg["backend"].dropna().unique()))

    def series(self) -> list[str]:
        if self.agg.empty:
            return []
        return sorted(set(self.agg.get("series", pd.Series(dtype=str)).dropna().unique()))

    def has(self, scenario: str) -> bool:
        return not self.agg.empty and scenario in set(self.agg.get("scenario", []))

    def scenario(self, name: str, level: str = "aggregate") -> pd.DataFrame:
        """Rows for one scenario (aggregated by default)."""
        source = self.agg if level == "aggregate" else self.raw
        if source.empty:
            return source
        subset = source[source.get("scenario") == name]
        return subset.copy()

    def concurrency_frame(self) -> pd.DataFrame:
        """Aggregate rows of the concurrency scenario, one per series/level."""
        frame = self.scenario("concurrency")
        if frame.empty:
            return frame
        keep = [
            c
            for c in (
                "series",
                "backend",
                "serving_mode",
                "concurrency",
                "throughput_tok_s",
                "ttft_ms_mean",
                "ttft_ms_p95",
                "ttft_ms_p99",
                "e2e_ms_mean",
                "e2e_ms_p99",
                "peak_memory_mb",
                "avg_gpu_util_pct",
                "num_requests",
                "run_id",
                "is_mock",
            )
            if c in frame.columns
        ]
        out = frame[keep].copy()
        out = self._collapse_runs(out, ["series", "concurrency"])
        if "concurrency" in out.columns:
            out = out.sort_values(["series", "concurrency"])
        return out

    def latency_frame(self) -> pd.DataFrame:
        frame = self.scenario("single")
        return self._collapse_runs(frame, ["series", "bucket", "concurrency"])

    def memory_frame(self) -> pd.DataFrame:
        frame = self.scenario("memory")
        return self._collapse_runs(frame, ["series", "bucket", "concurrency"])

    def prefix_frame(self) -> pd.DataFrame:
        frame = self.scenario("prefix_cache")
        return self._collapse_runs(frame, ["series", "group", "concurrency"])

    def scheduler_frame(self) -> pd.DataFrame:
        """Scheduler rows, de-duplicated across identical runs.

        The simulator is deterministic for a given seed, so combining several
        runs of the *same* simulation would otherwise plot each policy three
        times.
        """
        frame = self.scenario("scheduler")
        if frame.empty or "serving_mode" not in frame.columns:
            return frame
        # Prefer rows that carry the simulator's extra fields over the generic
        # aggregate rows, then keep one row per policy.
        if "extra_short_ttft_ms" in frame.columns:
            frame = frame.sort_values("extra_short_ttft_ms", ascending=False, na_position="last")
        return frame.drop_duplicates(subset=["serving_mode"], keep="first")

    # -- headline metrics -------------------------------------------------
    def kpis(self) -> dict[str, Any]:
        kpis: dict[str, Any] = {
            "runs": len(self.runs),
            "records": int(len(self.raw)),
            "backends": self.backends(),
            "is_mock": self.is_mock,
            "model": None,
            "gpu": None,
            "scenarios": sorted(set(self.agg.get("scenario", []))) if not self.agg.empty else [],
        }
        if self.runs:
            meta = self.runs[0].metadata
            kpis["model"] = meta.model_id
            kpis["gpu"] = meta.system.get("gpu_name")
            kpis["created_at"] = meta.created_at
            kpis["tags"] = meta.tags

        conc = self.concurrency_frame()
        if not conc.empty and "throughput_tok_s" in conc.columns:
            idx = conc["throughput_tok_s"].idxmax()
            kpis["peak_throughput_tok_s"] = _num(conc.loc[idx, "throughput_tok_s"])
            kpis["peak_throughput_backend"] = conc.loc[idx, "series"]
            kpis["peak_concurrency"] = _num(conc.loc[idx, "concurrency"])
            kpis["peak_util_pct"] = (
                _num(conc.loc[idx, "avg_gpu_util_pct"])
                if "avg_gpu_util_pct" in conc.columns
                else None
            )
            kpis["best_ttft_p95_ms"] = (
                _num(conc["ttft_ms_p95"].min()) if "ttft_ms_p95" in conc.columns else None
            )
            kpis["worst_e2e_p99_ms"] = (
                _num(conc["e2e_ms_p99"].max()) if "e2e_ms_p99" in conc.columns else None
            )

            # Throughput speed-up of the fastest series over the slowest one at
            # a shared concurrency level. Works for vllm-vs-naive and for
            # combined multi-run reports.
            pivot = conc.pivot_table(
                index="concurrency", columns="series", values="throughput_tok_s", aggfunc="max"
            ).dropna(how="all")
            if pivot.shape[1] >= 2:
                shared = pivot.dropna()
                if not shared.empty:
                    row = shared.iloc[-1]
                    series_max = row.max()
                    series_min = row.min()
                    kpis["throughput_speedup"] = round(safe_div(series_max, series_min, 0.0), 3)
                    kpis["throughput_speedup_at"] = _num(shared.index[-1])
                    kpis["speedup_fastest"] = str(row.idxmax())
                    kpis["speedup_baseline"] = str(row.idxmin())

        mem = self.memory_frame()
        if not mem.empty and "kv_cache_delta_mb" in mem.columns:
            valid = mem.dropna(subset=["kv_cache_delta_mb"])
            if not valid.empty:
                idx = valid["prompt_len"].idxmax()
                row = valid.loc[idx]
                kpis["kv_delta_mb"] = _num(row["kv_cache_delta_mb"])
                kpis["kv_delta_at_tokens"] = _num(row["prompt_len"])
                by_series = valid.groupby("series")["kv_cache_delta_mb"].max().dropna()
                if len(by_series) > 1:
                    best_series = by_series.idxmin()
                    worst_series = by_series.idxmax()
                    best_val = _num(by_series[best_series])
                    worst_val = _num(by_series[worst_series])
                    kpis["memory_saving_pct"] = round(
                        100.0 * (1.0 - safe_div(best_val, worst_val, 1.0)), 1
                    )
                    kpis["memory_best_backend"] = best_series
                    kpis["memory_baseline"] = worst_series

        prefix = self.prefix_frame()
        if not prefix.empty and "ttft_ms_mean" in prefix.columns:
            off = prefix[prefix["group"].astype(str).str.startswith("cache_off")]
            on = prefix[prefix["group"].astype(str).str.startswith("cache_on")]
            # Compare the last round only: that is where reuse is maximal.
            if not off.empty and not on.empty:
                off_last = off.sort_values("group").iloc[-1]
                on_last = on.sort_values("group").iloc[-1]
                off_ttft = _num(off_last["ttft_ms_mean"])
                on_ttft = _num(on_last["ttft_ms_mean"])
                if off_ttft and on_ttft:
                    kpis["prefix_speedup"] = round(off_ttft / on_ttft, 2)

        sched = self.scheduler_frame()
        if not sched.empty and "ttft_ms_p95" in sched.columns:
            valid = sched.dropna(subset=["ttft_ms_mean"])
            if not valid.empty:
                best = valid.loc[valid["ttft_ms_mean"].idxmin()]
                kpis["best_policy"] = best.get("serving_mode")
                kpis["best_policy_ttft_ms"] = _num(best["ttft_ms_mean"])
        return kpis

    # -- comparison table -------------------------------------------------
    def comparison(self) -> pd.DataFrame:
        """Side-by-side backend comparison at the highest shared concurrency."""
        conc = self.concurrency_frame()
        if conc.empty:
            return pd.DataFrame()
        top = conc["concurrency"].max()
        at_top = conc[conc["concurrency"] == top].copy()
        columns = {
            "series": "Series",
            "serving_mode": "Serving mode",
            "concurrency": "Concurrency",
            "throughput_tok_s": "Throughput (tok/s)",
            "ttft_ms_mean": "TTFT mean (ms)",
            "ttft_ms_p95": "TTFT p95 (ms)",
            "e2e_ms_p99": "E2E p99 (ms)",
            "avg_gpu_util_pct": "GPU util (%)",
            "peak_memory_mb": "Peak memory (MB)",
        }
        keep = [c for c in columns if c in at_top.columns]
        table = at_top[keep].rename(columns=columns).reset_index(drop=True)

        # Attach relative speed-up vs. the slowest backend in the table.
        if "Throughput (tok/s)" in table.columns and len(table) > 1:
            base = table["Throughput (tok/s)"].min()
            table["Speed-up"] = (table["Throughput (tok/s)"] / base).round(2)
        return table

    def scaling_efficiency(self) -> pd.DataFrame:
        """Throughput per unit concurrency — reveals saturation points."""
        conc = self.concurrency_frame()
        if conc.empty or "concurrency" not in conc.columns:
            return pd.DataFrame()
        conc = conc.dropna(subset=["throughput_tok_s", "concurrency"])
        conc = conc[conc["concurrency"] > 0]
        conc["efficiency_tok_s_per_stream"] = (
            conc["throughput_tok_s"] / conc["concurrency"]
        ).round(3)
        return conc

    def summary_records(self) -> list[dict[str, Any]]:
        return self.agg.to_dict("records") if not self.agg.empty else []
