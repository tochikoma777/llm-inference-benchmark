"""Run scheduler-policy comparisons and translate results into records.

The simulator is deterministic given a seed, so every policy sees the exact
same workload — the only thing that changes is the scheduling decision.
"""

from __future__ import annotations

from ..core.config import RunConfig
from ..core.logging import get_logger
from .engine import Simulation
from .model import generate_workload
from .policies import get_policy

__all__ = ["run_scheduler_comparison", "compare_policies"]

log = get_logger("scheduler.runner")


def compare_policies(config: RunConfig) -> list:
    """Run every configured policy over one shared workload."""
    sim_cfg = config.sim
    workload = generate_workload(
        num_requests=sim_cfg.num_requests,
        short_ratio=sim_cfg.short_ratio,
        mean_interarrival_ms=sim_cfg.arrival_mean_ms,
        seed=sim_cfg.seed,
    )
    results = []
    for name in sim_cfg.policies:
        policy = get_policy(
            name, **({"aging_tau_ms": sim_cfg.sjf_aging_tau_ms} if name == "sjf_aging" else {})
        )
        simulation = Simulation(sim_cfg, policy)
        result = simulation.run(workload)
        log.info(
            "policy=%-11s ttft_p95=%7.1fms e2e_p99=%8.1fms thr=%7.1f tok/s preempt=%d",
            result.label,
            result.metrics["ttft_p95_ms"],
            result.metrics["e2e_p99_ms"],
            result.metrics["throughput_tok_s"],
            int(result.metrics["preemptions"]),
        )
        results.append(result)
    return results


def run_scheduler_comparison(config: RunConfig) -> list[dict]:
    """Execute the comparison and return flat records for the result store."""
    records: list[dict] = []
    for result in compare_policies(config):
        for request in result.requests:
            records.append(
                {
                    "scenario": "scheduler",
                    "level": "request",
                    "backend": "simulation",
                    "serving_mode": result.policy,
                    "group": request.group,
                    "bucket": result.policy,
                    "concurrency": config.sim.max_num_seqs,
                    "request_index": request.id,
                    "prompt_len": request.prompt_len,
                    "input_tokens": request.prompt_len,
                    "output_len": request.output_len,
                    "output_tokens": request.generated,
                    "arrival_ms": round(request.arrival_ms, 3),
                    "start_ms": round(request.start_ms or 0.0, 3),
                    "queue_ms": round(request.wait_ms, 3),
                    "ttft_ms": (
                        round(request.ttft_latency_ms, 3)
                        if request.ttft_latency_ms is not None
                        else None
                    ),
                    "e2e_ms": round(request.e2e_ms, 3) if request.e2e_ms is not None else None,
                    "finish_ms": round(request.finish_ms, 3) if request.finish_ms else None,
                    "total_time_ms": round(request.e2e_ms, 3) if request.e2e_ms else None,
                    "throughput_tok_s": None,
                    "success": True,
                    "extra_policy_label": result.label,
                    "extra_preemptions": request.preemptions,
                    "extra_job_group": request.group,
                }
            )

        metrics = result.metrics
        records.append(
            {
                "scenario": "scheduler",
                "level": "aggregate",
                "backend": "simulation",
                "serving_mode": result.policy,
                "group": result.policy,
                "bucket": result.policy,
                "concurrency": config.sim.max_num_seqs,
                "num_requests": metrics["num_requests"],
                "output_tokens": metrics["output_tokens"],
                "makespan_ms": round(metrics["makespan_ms"], 3),
                "throughput_tok_s": round(metrics["throughput_tok_s"], 3),
                "ttft_ms": round(metrics["ttft_mean_ms"], 3),
                "ttft_ms_mean": round(metrics["ttft_mean_ms"], 3),
                "ttft_ms_p50": round(metrics["ttft_p50_ms"], 3),
                "ttft_ms_p95": round(metrics["ttft_p95_ms"], 3),
                "ttft_ms_p99": round(metrics["ttft_p99_ms"], 3),
                "e2e_ms": round(metrics["e2e_mean_ms"], 3),
                "e2e_ms_mean": round(metrics["e2e_mean_ms"], 3),
                "e2e_ms_p95": round(metrics["e2e_p95_ms"], 3),
                "e2e_ms_p99": round(metrics["e2e_p99_ms"], 3),
                "queue_ms": round(metrics["wait_mean_ms"], 3),
                "queue_ms_mean": round(metrics["wait_mean_ms"], 3),
                "queue_ms_max": round(metrics["wait_max_ms"], 3),
                "success": True,
                "extra_policy_label": result.label,
                "extra_policy_description": result.description,
                "extra_short_ttft_ms": round(metrics["short_ttft_mean_ms"], 3),
                "extra_long_ttft_ms": round(metrics["long_ttft_mean_ms"], 3),
                "extra_short_e2e_ms": round(metrics["short_e2e_mean_ms"], 3),
                "extra_long_e2e_ms": round(metrics["long_e2e_mean_ms"], 3),
                "extra_preemptions": int(metrics["preemptions"]),
                "extra_fairness_jain": round(metrics["fairness_jain"], 4),
            }
        )
    return records
