"""Configuration loading, overrides and validation."""

from __future__ import annotations

import json

import pytest
import yaml

from llmbench.core.config import (
    ConfigError,
    RunConfig,
    ValidationError,
    load_config,
)


def test_defaults_are_valid():
    config = load_config()
    assert isinstance(config, RunConfig)
    assert config.backend.name in {"mock", "naive", "vllm"}
    assert config.workload.max_new_tokens >= 1


def test_load_yaml_file(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "name": "custom",
                "backend": {"name": "vllm", "serving_mode": "offline"},
                "workload": {"num_requests": 7, "arrival": {"mode": "poisson"}},
            }
        ),
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.name == "custom"
    assert config.backend.name == "vllm"
    assert config.backend.effective_serving_mode == "offline"
    assert config.workload.num_requests == 7
    assert config.workload.arrival.mode == "poisson"
    # untouched fields keep their defaults
    assert config.workload.max_new_tokens == 128


def test_load_json_file(tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"name": "json-run", "seed": 11}), encoding="utf-8")
    config = load_config(path)
    assert config.name == "json-run"
    assert config.seed == 11


def test_dotted_overrides_coerce_types():
    config = load_config(
        overrides=[
            "backend.name=mock",
            "scenarios.concurrency_levels=[1,2,4]",
            "workload.temperature=0.0",
            "monitor.enabled=false",
        ]
    )
    assert config.backend.name == "mock"
    assert config.scenarios.concurrency_levels == [1, 2, 4]
    assert config.workload.temperature == 0.0
    assert config.monitor.enabled is False


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("LLMBENCH_BACKEND", "naive")
    monkeypatch.setenv("LLMBENCH_SEED", "99")
    config = load_config()
    assert config.backend.name == "naive"
    assert config.seed == 99


def test_missing_file_raises(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "nope.yaml")


def test_malformed_yaml_raises(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("name: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


def test_top_level_must_be_mapping(tmp_path):
    path = tmp_path / "list.yaml"
    path.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


def test_validation_rejects_unknown_backend():
    with pytest.raises(ValidationError):
        load_config(overrides={"backend.name": "tensorrt"})


def test_validation_rejects_bad_concurrency():
    with pytest.raises(ValidationError):
        load_config(overrides={"scenarios.concurrency_levels": [1, 0]})


def test_validation_rejects_bad_arrival_mode():
    with pytest.raises(ValidationError):
        load_config(overrides={"workload.arrival.mode": "weird"})


def test_validation_rejects_bad_utilization():
    with pytest.raises(ValidationError):
        load_config(overrides={"backend.gpu_memory_utilization": 1.5})


def test_unknown_keys_are_preserved_in_extra(tmp_path):
    path = tmp_path / "extra.yaml"
    path.write_text(
        yaml.safe_dump({"name": "x", "custom_flag": 123, "backend": {"api_key": "abc"}}),
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.extra["custom_flag"] == 123
    assert config.backend.extra["api_key"] == "abc"


def test_roundtrip_to_dict_and_save(tmp_path):
    config = load_config(overrides={"name": "roundtrip"})
    out = config.save(tmp_path / "saved.yaml")
    reloaded = load_config(out)
    assert reloaded.name == "roundtrip"
    assert reloaded.to_dict()["backend"]["name"] == config.backend.name


def test_auto_serving_mode_resolution():
    config = load_config(overrides={"backend.name": "naive", "backend.serving_mode": "auto"})
    assert config.backend.effective_serving_mode == "static_batch"
    config = load_config(overrides={"backend.name": "vllm", "backend.serving_mode": "auto"})
    assert config.backend.effective_serving_mode == "continuous_batch"


def test_model_path_resolution_error():
    config = load_config(overrides={"model.path": "/definitely/not/here"})
    with pytest.raises(ConfigError) as excinfo:
        config.model.resolve_path()
    assert "does not exist" in str(excinfo.value)


def test_sim_policies_validation():
    with pytest.raises(ValidationError):
        load_config(overrides={"sim.policies": ["fcfs", "magic"]})
