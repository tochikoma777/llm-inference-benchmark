"""Iteration-level simulation engine.

Each simulated iteration mirrors one vLLM scheduler step:

1. admit newly arrived requests, subject to ``max_num_seqs`` and free blocks
2. prefill all newly admitted prompts (their first token is produced here)
3. decode one token for every already-running sequence
4. charge the iteration wall-time: prefill tokens + batch-dependent decode cost

Memory exhaustion triggers preemption (the victim's KV blocks are released and
it must recompute later), which is how the real engine behaves.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from ..core.config import SimConfig
from ..core.logging import get_logger
from ..metrics.stats import percentile, safe_div
from .model import BlockPool, SimRequest
from .policies import Policy

__all__ = ["PolicyResult", "Simulation"]

log = get_logger("scheduler.engine")


@dataclass
class PolicyResult:
    policy: str
    label: str
    description: str
    requests: list[SimRequest]
    makespan_ms: float
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def completed(self) -> int:
        return sum(1 for r in self.requests if r.finish_ms is not None)


def _jain_index(values: Sequence[float]) -> float:
    """Jain's fairness index; 1.0 is perfectly fair."""
    squares = [v * v for v in values]
    denom = len(squares) * sum(squares)
    if denom <= 0:
        return 0.0
    return (sum(values) ** 2) / denom


class Simulation:
    """Runs one workload against one policy."""

    def __init__(self, config: SimConfig, policy: Policy):
        self.config = config
        self.policy = policy
        self.pool = BlockPool(
            total_mb=config.kv_pool_mb,
            block_size=config.block_size,
            kv_mb_per_token=config.kv_mb_per_token,
        )
        self.preemptions = 0
        self.iterations = 0
        self.dropped: list[SimRequest] = []
        #: A request preempted more often than this can never make progress
        #: (its KV footprint exceeds the pool) — drop it instead of looping.
        self.max_preemptions = 8

    # -- main loop -------------------------------------------------------
    def run(self, workload: Sequence[SimRequest]) -> PolicyResult:
        self.iterations = 0
        # Work on copies so the same workload can be replayed for each policy.
        requests = [
            SimRequest(
                id=r.id,
                arrival_ms=r.arrival_ms,
                prompt_len=r.prompt_len,
                output_len=r.output_len,
                priority=r.priority,
                group=r.group,
            )
            for r in workload
        ]
        requests.sort(key=lambda r: (r.arrival_ms, r.id))

        cfg = self.config
        pool = self.pool
        waiting: list[SimRequest] = []
        running: list[SimRequest] = []
        finished: list[SimRequest] = []

        now = 0.0
        cursor = 0
        guard = max(len(requests) * 500, 100_000)

        while len(finished) + len(self.dropped) < len(requests) and self.iterations < guard:
            self.iterations += 1

            # 1) arrivals
            while cursor < len(requests) and requests[cursor].arrival_ms <= now:
                waiting.append(requests[cursor])
                cursor += 1
            if not running and not waiting and cursor < len(requests):
                now = max(requests[cursor].arrival_ms, now)
                continue

            # 2) Admission. Following vLLM's default policy, waiting requests
            #    never preempt running ones: if a prompt does not fit, it waits
            #    for memory to be released. This also prevents the livelock in
            #    which two requests preempt each other forever.
            while len(running) < cfg.max_num_seqs and waiting:
                progressed = False
                for candidate in self.policy.order(waiting, now):
                    need_tokens = candidate.required_tokens + 1
                    if need_tokens > pool.capacity_tokens:
                        # Can never fit, even in an empty pool.
                        waiting.remove(candidate)
                        if candidate not in self.dropped:
                            self.dropped.append(candidate)
                        log.warning(
                            "Dropping request %d: needs %d tokens but the KV pool " "holds %d",
                            candidate.id,
                            need_tokens,
                            pool.capacity_tokens,
                        )
                        progressed = True
                        break
                    if pool.ensure(candidate.id, need_tokens):
                        candidate.start_ms = now
                        running.append(candidate)
                        waiting.remove(candidate)
                        progressed = True
                        break
                    break  # memory is full: stop admitting
                if not progressed:
                    break

            if not running:
                # Nothing is running. Either wait for the next arrival, or bail
                # out if even a completely empty pool cannot admit anything.
                if cursor >= len(requests) and not waiting:
                    break
                if waiting and pool.free_blocks == pool.total_blocks:
                    log.warning("No waiting request fits in the KV pool; aborting simulation.")
                    break
                if cursor < len(requests):
                    now = max(requests[cursor].arrival_ms, now + 1.0)
                else:
                    now += 1.0
                continue

            # 3) iteration cost
            prefill_tokens = 0
            decoding = 0
            for req in running:
                deficit = req.required_tokens - req.computed_tokens
                if deficit > 0:
                    prefill_tokens += deficit
                else:
                    decoding += 1
            dt = prefill_tokens / cfg.prefill_tok_per_ms
            if decoding:
                dt += cfg.decode_step_ms * (1.0 + (decoding - 1) * cfg.decode_batch_factor)
            now += max(dt, 1e-9)

            # 4) progress
            for req in list(running):
                if req.computed_tokens < req.required_tokens:
                    # Prefill (or recompute after preemption). Only the very
                    # first prefill emits a token; a recompute replays tokens
                    # that were already counted.
                    first_prefill = req.ttft_ms is None
                    req.computed_tokens = req.required_tokens
                    if first_prefill:
                        req.ttft_ms = now
                        req.generated += 1
                        req.computed_tokens = req.required_tokens
                else:
                    req.generated += 1
                    req.computed_tokens += 1

                if req.done:
                    req.finish_ms = now
                    pool.release(req.id)
                    running.remove(req)
                    finished.append(req)
                    continue

                if not pool.ensure(req.id, req.computed_tokens):
                    # Allocation failed mid-decode: preempt ourselves.
                    self._preempt(req, pool, waiting, running)

        if len(finished) + len(self.dropped) < len(requests):
            log.warning(
                "Simulation stopped early: %d/%d requests completed (%d dropped)",
                len(finished),
                len(requests),
                len(self.dropped),
            )

        makespan = 0.0
        if finished:
            finish_times = [r.finish_ms for r in finished if r.finish_ms is not None]
            if finish_times:
                makespan = max(finish_times) - min(r.arrival_ms for r in requests)
        return PolicyResult(
            policy=self.policy.name,
            label=self.policy.label,
            description=self.policy.description,
            requests=finished,
            makespan_ms=makespan,
            metrics=self._metrics(finished, makespan),
        )

    # -- helpers ---------------------------------------------------------
    def _preempt(
        self,
        victim: SimRequest,
        pool: BlockPool,
        waiting: list[SimRequest],
        running: list[SimRequest] | None = None,
    ) -> None:
        """Swap a request out: free its blocks and move it back to waiting."""
        pool.release(victim.id)
        victim.computed_tokens = 0
        victim.preemptions += 1
        victim.start_ms = None
        self.preemptions += 1
        if running is not None:
            while victim in running:
                running.remove(victim)
        if victim not in waiting:
            waiting.append(victim)

        if victim.preemptions > self.max_preemptions:
            # Infeasible configuration: the sequence never fits together with
            # everything else. Drop it rather than thrash forever.
            if victim in waiting:
                waiting.remove(victim)
            if victim not in self.dropped:
                self.dropped.append(victim)
            log.warning(
                "Dropping request %d after %d preemptions (KV pool too small)",
                victim.id,
                victim.preemptions,
            )

    def _metrics(self, finished: list[SimRequest], makespan_ms: float) -> dict[str, float]:
        ttft = [r.ttft_latency_ms for r in finished if r.ttft_latency_ms is not None]
        e2e = [r.e2e_ms for r in finished if r.e2e_ms is not None]
        wait = [r.wait_ms for r in finished]
        tokens = sum(r.generated for r in finished)

        short = [r for r in finished if r.group == "short"]
        long_ = [r for r in finished if r.group == "long"]

        def _avg(values: Sequence[float | None]) -> float:
            clean = [v for v in values if v is not None]
            return sum(clean) / len(clean) if clean else 0.0

        return {
            "num_requests": len(finished),
            "makespan_ms": makespan_ms,
            "output_tokens": tokens,
            "throughput_tok_s": safe_div(tokens * 1000.0, makespan_ms, 0.0),
            "ttft_mean_ms": _avg(ttft),
            "ttft_p50_ms": percentile(ttft, 50) or 0.0,
            "ttft_p95_ms": percentile(ttft, 95) or 0.0,
            "ttft_p99_ms": percentile(ttft, 99) or 0.0,
            "e2e_mean_ms": _avg(e2e),
            "e2e_p50_ms": percentile(e2e, 50) or 0.0,
            "e2e_p95_ms": percentile(e2e, 95) or 0.0,
            "e2e_p99_ms": percentile(e2e, 99) or 0.0,
            "wait_mean_ms": _avg(wait),
            "wait_max_ms": max(wait) if wait else 0.0,
            "short_ttft_mean_ms": _avg([r.ttft_latency_ms for r in short]),
            "long_ttft_mean_ms": _avg([r.ttft_latency_ms for r in long_]),
            "short_e2e_mean_ms": _avg([r.e2e_ms for r in short]),
            "long_e2e_mean_ms": _avg([r.e2e_ms for r in long_]),
            "preemptions": float(self.preemptions),
            "dropped": float(len(self.dropped)),
            "fairness_jain": _jain_index([safe_div(1.0, r.e2e_ms or 0.0, 0.0) for r in finished]),
        }
