"""Core entities of the scheduler simulator.

The simulation is intentionally conservative: it models the *scheduling*
decisions (admission order, batch composition, block allocation, preemption)
rather than kernel-level performance. That is what makes FCFS-vs-SJF
comparisons meaningful and cheap enough to run on a laptop.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

__all__ = ["SimRequest", "BlockPool", "generate_workload"]


@dataclass
class SimRequest:
    """A request travelling through the simulated serving pipeline."""

    id: int
    arrival_ms: float
    prompt_len: int
    output_len: int
    priority: int = 0
    group: str = "short"

    # -- runtime state ---------------------------------------------------
    computed_tokens: int = 0  # tokens whose KV is resident on the GPU
    generated: int = 0  # output tokens produced so far
    start_ms: float | None = None
    ttft_ms: float | None = None
    finish_ms: float | None = None
    preemptions: int = 0
    blocks: int = 0

    @property
    def total_len(self) -> int:
        return self.prompt_len + self.output_len

    @property
    def required_tokens(self) -> int:
        """Tokens that must be resident before the next decode step."""
        return self.prompt_len + self.generated

    @property
    def done(self) -> bool:
        return self.generated >= self.output_len

    @property
    def e2e_ms(self) -> float | None:
        if self.finish_ms is None:
            return None
        return self.finish_ms - self.arrival_ms

    @property
    def ttft_latency_ms(self) -> float | None:
        if self.ttft_ms is None:
            return None
        return self.ttft_ms - self.arrival_ms

    @property
    def wait_ms(self) -> float:
        if self.start_ms is None:
            return 0.0
        return max(self.start_ms - self.arrival_ms, 0.0)


class BlockPool:
    """Paged KV-cache allocator (fixed-size blocks, on-demand growth).

    Unlike the classic implementation this pool does **not** reserve
    ``prompt_len + output_len`` up front — that is precisely the behaviour
    PagedAttention replaced. Blocks are granted as the sequence grows, so a
    request that finishes early returns memory to the pool.
    """

    def __init__(self, total_mb: float, block_size: int, kv_mb_per_token: float):
        self.block_size = max(int(block_size), 1)
        self.block_mb = max(self.block_size * float(kv_mb_per_token), 1e-9)
        self.total_blocks = max(int(total_mb // self.block_mb), 1)
        self.free_blocks = self.total_blocks
        self._allocated: dict[int, int] = {}

    # -- queries ---------------------------------------------------------
    def blocks_for(self, tokens: int) -> int:
        return (max(tokens, 0) + self.block_size - 1) // self.block_size

    @property
    def capacity_tokens(self) -> int:
        """Largest sequence the pool could ever hold on its own."""
        return self.total_blocks * self.block_size

    @property
    def utilization(self) -> float:
        return 1.0 - (self.free_blocks / self.total_blocks)

    def allocated_blocks(self, request_id: int) -> int:
        return self._allocated.get(request_id, 0)

    # -- mutations -------------------------------------------------------
    def ensure(self, request_id: int, tokens: int) -> bool:
        """Grow the allocation so that ``tokens`` fit. Returns success."""
        needed = self.blocks_for(tokens)
        current = self._allocated.get(request_id, 0)
        delta = needed - current
        if delta <= 0:
            return True
        if delta > self.free_blocks:
            return False
        self._allocated[request_id] = needed
        self.free_blocks -= delta
        return True

    def release(self, request_id: int) -> None:
        held = self._allocated.pop(request_id, 0)
        self.free_blocks += held

    def can_fit(self, tokens: int) -> bool:
        return self.blocks_for(tokens) <= self.free_blocks

    def snapshot(self) -> dict[str, float]:
        return {
            "total_blocks": self.total_blocks,
            "free_blocks": self.free_blocks,
            "utilization": self.utilization,
        }


def generate_workload(
    num_requests: int = 200,
    short_ratio: float = 0.7,
    mean_interarrival_ms: float = 80.0,
    seed: int = 42,
) -> list[SimRequest]:
    """Generate a mixed short/long workload with Poisson arrivals.

    Short requests model FAQ/chat traffic; long requests model document
    generation and code completion. The mix is what makes SJF look attractive
    and its tail-latency cost visible.
    """
    rng = random.Random(seed)
    requests: list[SimRequest] = []
    now = 0.0
    for i in range(num_requests):
        now += rng.expovariate(1.0 / max(mean_interarrival_ms, 1e-6))
        if rng.random() < short_ratio:
            prompt = rng.randint(50, 256)
            output = rng.randint(20, 128)
            group = "short"
            priority = 5
        else:
            prompt = rng.randint(512, 2048)
            output = rng.randint(256, 1024)
            group = "long"
            priority = 1
        requests.append(
            SimRequest(
                id=i,
                arrival_ms=round(now, 3),
                prompt_len=prompt,
                output_len=output,
                priority=priority,
                group=group,
            )
        )
    return requests
