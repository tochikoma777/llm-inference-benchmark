"""Result storage: writing, reading, aggregation."""

from __future__ import annotations

import json

import pytest

from llmbench.core.results import (
    RECORD_DEFAULTS,
    ResultStore,
    RunMetadata,
    RunWriter,
    aggregate_records,
    new_run_id,
    normalize_record,
)


def _record(**overrides):
    base = {
        "scenario": "concurrency",
        "level": "request",
        "backend": "vllm",
        "serving_mode": "continuous_batch",
        "bucket": "load",
        "concurrency": 4,
        "prompt_len": 256,
        "output_len": 64,
        "output_tokens": 64,
        "arrival_ms": 0.0,
        "start_ms": 0.0,
        "ttft_ms": 100.0,
        "e2e_ms": 1100.0,
        "finish_ms": 1100.0,
        "success": True,
    }
    base.update(overrides)
    return base


def test_normalize_fills_canonical_columns():
    rec = normalize_record({"scenario": "single"})
    for key in RECORD_DEFAULTS:
        assert key in rec
    assert rec["scenario"] == "single"


def test_run_id_is_sortable_and_unique():
    first = new_run_id("My Run")
    second = new_run_id("My Run")
    assert first != second
    assert "my-run" in first
    assert first[:8].isdigit()


def test_writer_creates_expected_files(tmp_path):
    meta = RunMetadata(run_id="r1", name="r1", backend="vllm")
    writer = RunWriter(meta, directory=tmp_path / "r1")
    writer.add(_record())
    path = writer.write()

    assert (path / "results.jsonl").exists()
    assert (path / "metadata.json").exists()
    assert (path / "summary.json").exists()

    meta_json = json.loads((path / "metadata.json").read_text())
    assert meta_json["status"] == "completed"
    assert meta_json["finished_at"] is not None


def test_writer_context_manager_marks_failure(tmp_path):
    meta = RunMetadata(run_id="r2", name="r2")
    with pytest.raises(RuntimeError), RunWriter(meta, directory=tmp_path / "r2"):
        raise RuntimeError("boom")
    saved = json.loads((tmp_path / "r2" / "metadata.json").read_text())
    assert saved["status"] == "failed"
    assert "boom" in saved["error"]


def test_records_get_ids_and_run_id(tmp_path):
    meta = RunMetadata(run_id="r3", name="r3")
    writer = RunWriter(meta, directory=tmp_path / "r3")
    writer.add(_record())
    writer.add(_record())
    ids = [rec["record_id"] for rec in writer.records]
    assert ids == ["r000000", "r000001"]
    assert all(rec["run_id"] == "r3" for rec in writer.records)


def test_aggregate_computes_makespan_throughput():
    records = [
        _record(ttft_ms=100, e2e_ms=1000, finish_ms=1000, request_index=0),
        _record(ttft_ms=200, e2e_ms=2000, finish_ms=2000, request_index=1),
    ]
    rows = aggregate_records(records)
    assert len(rows) == 1
    row = rows[0]
    assert row["num_requests"] == 2
    assert row["output_tokens"] == 128
    # makespan = 2000 ms → 128 tokens / 2 s
    assert row["throughput_tok_s"] == pytest.approx(64.0)
    assert row["ttft_ms_mean"] == pytest.approx(150.0)
    assert row["ttft_ms_p95"] is not None


def test_aggregate_never_merges_different_runs():
    records = [
        _record(run_id="run-a", e2e_ms=1000, finish_ms=1000),
        _record(run_id="run-b", e2e_ms=1000, finish_ms=1000),
    ]
    rows = aggregate_records(records)
    assert len(rows) == 2
    # Each run contributes 64 tokens over 1 s — not 128 tokens.
    assert all(row["throughput_tok_s"] == pytest.approx(64.0) for row in rows)


def test_aggregate_respects_buckets():
    records = [
        _record(bucket="load", concurrency=1, e2e_ms=1000, finish_ms=1000),
        _record(bucket="load", concurrency=4, e2e_ms=1000, finish_ms=1000),
        _record(bucket="other", concurrency=4, e2e_ms=1000, finish_ms=1000),
    ]
    rows = aggregate_records(records)
    assert len(rows) == 3


def test_aggregate_skips_failures_and_aggregate_rows():
    records = [
        _record(),
        _record(success=False),
        _record(level="aggregate"),
    ]
    rows = aggregate_records(records)
    assert rows[0]["num_requests"] == 1


def test_store_roundtrip(tmp_path):
    meta = RunMetadata(run_id="roundtrip", name="rt", backend="vllm")
    writer = RunWriter(meta, directory=tmp_path / "roundtrip")
    writer.add(_record())
    writer.write()

    store = ResultStore(tmp_path)
    runs = store.runs()
    assert len(runs) == 1
    assert runs[0].run_id == "roundtrip"
    assert runs[0].num_records == 1

    records = store.load_records("roundtrip")
    assert records[0]["ttft_ms"] == 100.0

    summary = store.summary("roundtrip")
    assert summary[0]["throughput_tok_s"] > 0


def test_store_load_all_returns_dataframe(synthetic_store):
    store = ResultStore(synthetic_store)
    df, infos = store.load_all()
    assert len(infos) == 2
    assert not df.empty
    assert {"run_id", "run_name", "scenario", "backend"} <= set(df.columns)
    assert set(df["run_id"].unique()) == {"aaa-vllm", "bbb-naive"}


def test_store_load_all_empty(tmp_path):
    store = ResultStore(tmp_path)
    df, infos = store.load_all()
    assert df.empty
    assert infos == []


def test_store_export_csv(synthetic_store, tmp_path):
    store = ResultStore(synthetic_store)
    out = store.export_csv(tmp_path / "export.csv")
    assert out.exists()
    assert "scenario" in out.read_text().splitlines()[0]


def test_store_delete(synthetic_store):
    store = ResultStore(synthetic_store)
    store.delete("bbb-naive")
    assert [info.run_id for info in store.runs()] == ["aaa-vllm"]


def test_store_skips_corrupted_run(tmp_path):
    bad = tmp_path / "broken"
    bad.mkdir()
    (bad / "metadata.json").write_text("{not json", encoding="utf-8")
    store = ResultStore(tmp_path)
    assert store.runs() == []
