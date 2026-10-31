"""KV-cache block allocator and scheduling policies."""

from __future__ import annotations

import pytest

from llmbench.core.config import SimConfig
from llmbench.scheduler_sim.engine import Simulation, _jain_index
from llmbench.scheduler_sim.model import BlockPool, SimRequest, generate_workload
from llmbench.scheduler_sim.policies import POLICIES, get_policy


# ---------------------------------------------------------------------------
# BlockPool
# ---------------------------------------------------------------------------
def test_blocks_for_rounds_up():
    pool = BlockPool(total_mb=100, block_size=16, kv_mb_per_token=1.0)
    assert pool.blocks_for(0) == 0
    assert pool.blocks_for(1) == 1
    assert pool.blocks_for(16) == 1
    assert pool.blocks_for(17) == 2


def test_pool_allocates_grows_and_releases():
    pool = BlockPool(total_mb=16 * 4, block_size=16, kv_mb_per_token=1.0)
    total = pool.total_blocks
    assert pool.ensure(1, 16) is True
    assert pool.allocated_blocks(1) == 1
    assert pool.ensure(1, 32) is True
    assert pool.allocated_blocks(1) == 2
    # Shrinking requests do not free blocks implicitly.
    assert pool.ensure(1, 16) is True
    assert pool.allocated_blocks(1) == 2
    pool.release(1)
    assert pool.free_blocks == total
    assert pool.utilization == 0.0


def test_pool_refuses_overcommit():
    pool = BlockPool(total_mb=16, block_size=16, kv_mb_per_token=1.0)
    assert pool.total_blocks == 1
    assert pool.ensure(1, 16) is True
    assert pool.ensure(2, 16) is False
    assert pool.can_fit(16) is False


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------
def _req(i, arrival, prompt, output, priority=0):
    return SimRequest(
        id=i, arrival_ms=arrival, prompt_len=prompt, output_len=output, priority=priority
    )


def test_fcfs_orders_by_arrival():
    waiting = [_req(1, 20, 100, 10), _req(0, 10, 900, 10)]
    ordered = get_policy("fcfs").order(waiting, now_ms=100)
    assert [r.id for r in ordered] == [0, 1]


def test_sjf_orders_by_total_length():
    waiting = [_req(0, 10, 900, 100), _req(1, 20, 10, 5)]
    ordered = get_policy("sjf").order(waiting, now_ms=100)
    assert [r.id for r in ordered] == [1, 0]


def test_priority_orders_by_priority_then_arrival():
    waiting = [_req(0, 10, 100, 10, priority=1), _req(1, 20, 100, 10, priority=9)]
    ordered = get_policy("priority").order(waiting, now_ms=100)
    assert [r.id for r in ordered] == [1, 0]


def test_aging_prefers_starved_long_job():
    fresh = _req(0, 10_000, 900, 100)
    starved = _req(1, 0, 900, 100)
    ordered = get_policy("sjf_aging", aging_tau_ms=1000).order([fresh, starved], now_ms=10_000)
    assert ordered[0].id == 1


def test_all_policies_registered():
    assert set(POLICIES) == {"fcfs", "sjf", "priority", "sjf_aging", "round_robin"}


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def _sim_config(**overrides) -> SimConfig:
    config = SimConfig(num_requests=30, max_num_seqs=4, kv_pool_mb=2000.0)
    for key, value in overrides.items():
        setattr(config, key, value)
    config.validate()
    return config


def test_workload_is_deterministic():
    first = generate_workload(10, seed=3)
    second = generate_workload(10, seed=3)
    assert [(r.arrival_ms, r.prompt_len, r.output_len) for r in first] == [
        (r.arrival_ms, r.prompt_len, r.output_len) for r in second
    ]


@pytest.mark.parametrize("policy", ["fcfs", "sjf", "priority", "sjf_aging", "round_robin"])
def test_simulation_completes_every_request(policy):
    workload = generate_workload(25, seed=1)
    result = Simulation(_sim_config(), get_policy(policy)).run(workload)
    assert result.completed == 25
    assert result.makespan_ms > 0
    assert result.metrics["throughput_tok_s"] > 0
    for request in result.requests:
        assert request.finish_ms is not None
        assert request.ttft_ms is not None
        assert request.generated == request.output_len


def test_ttft_is_never_before_arrival():
    workload = generate_workload(15, seed=5)
    result = Simulation(_sim_config(), get_policy("fcfs")).run(workload)
    for request in result.requests:
        assert request.ttft_latency_ms >= 0
        assert request.e2e_ms >= request.ttft_latency_ms


def test_sjf_improves_short_job_ttft():
    workload = generate_workload(40, seed=7, short_ratio=0.7)
    config = _sim_config(max_num_seqs=2)
    fcfs = Simulation(config, get_policy("fcfs")).run(workload)
    sjf = Simulation(config, get_policy("sjf")).run(workload)
    assert sjf.metrics["short_ttft_mean_ms"] < fcfs.metrics["short_ttft_mean_ms"]
    # ... at the cost of the long tail.
    assert sjf.metrics["long_ttft_mean_ms"] >= fcfs.metrics["long_ttft_mean_ms"]


def test_preemption_happens_when_memory_is_tight():
    workload = generate_workload(30, seed=2)
    config = _sim_config(max_num_seqs=8, kv_pool_mb=2600.0, kv_mb_per_token=0.5)
    result = Simulation(config, get_policy("fcfs")).run(workload)
    assert result.metrics["preemptions"] > 0
    assert result.completed == 30


def test_tiny_pool_terminates_without_completing():
    workload = generate_workload(10, seed=3)
    config = _sim_config(kv_pool_mb=0.5, kv_mb_per_token=10.0)
    result = Simulation(config, get_policy("fcfs")).run(workload)
    assert result.completed == 0


def test_request_that_cannot_grow_is_dropped():
    # Room for the 100-token prompt but not for the 400-token output.
    workload = [
        SimRequest(id=0, arrival_ms=0.0, prompt_len=100, output_len=400),
        SimRequest(id=1, arrival_ms=1.0, prompt_len=100, output_len=10),
    ]
    config = _sim_config(kv_pool_mb=60.0, kv_mb_per_token=0.5, max_num_seqs=1)
    result = Simulation(config, get_policy("fcfs")).run(workload)
    assert result.metrics["dropped"] >= 1
    assert all(r.finish_ms is not None for r in result.requests)


def test_jain_index_bounds():
    assert _jain_index([1, 1, 1, 1]) == pytest.approx(1.0)
    assert 0 < _jain_index([1, 10, 100]) < 1
    assert _jain_index([]) == 0


def test_metrics_are_self_consistent():
    workload = generate_workload(30, seed=11)
    result = Simulation(_sim_config(), get_policy("fcfs")).run(workload)
    metrics = result.metrics
    assert metrics["ttft_p50_ms"] <= metrics["ttft_p95_ms"] <= metrics["ttft_p99_ms"]
    assert metrics["e2e_mean_ms"] <= metrics["e2e_p99_ms"]
    assert metrics["num_requests"] == 30
