"""Scheduling policies for the simulator.

Every policy only decides *ordering*; admission limits and memory safety are
enforced by the engine. This keeps policies small and comparable.
"""

from __future__ import annotations

import abc
from collections.abc import Sequence

from .model import SimRequest

__all__ = [
    "Policy",
    "FCFSPolicy",
    "SJfPolicy",
    "PriorityPolicy",
    "AgingSJfPolicy",
    "RoundRobinPolicy",
    "POLICIES",
    "get_policy",
]


class Policy(abc.ABC):
    """Ordering strategy for the waiting queue."""

    name: str = "policy"
    label: str = "Policy"
    description: str = ""

    @abc.abstractmethod
    def order(self, waiting: Sequence[SimRequest], now_ms: float) -> list[SimRequest]:
        """Return ``waiting`` sorted best-first."""

    def preempt_candidate(
        self, running: Sequence[SimRequest], newcomer: SimRequest
    ) -> SimRequest | None:
        """Choose a victim when memory is exhausted (largest remaining work)."""
        candidates = [r for r in running if r.id != newcomer.id]
        if not candidates:
            return None
        return max(candidates, key=lambda r: (r.total_len - r.generated, r.id))

    def as_dict(self) -> dict[str, str]:
        return {"name": self.name, "label": self.label, "description": self.description}


class FCFSPolicy(Policy):
    name = "fcfs"
    label = "FCFS"
    description = "First come, first served — fair, but head-of-line blocking."

    def order(self, waiting, now_ms):
        return sorted(waiting, key=lambda r: (r.arrival_ms, r.id))


class SJfPolicy(Policy):
    """Shortest-job-first using the true job size (an upper bound on SJF gain)."""

    name = "sjf"
    label = "SJF (oracle)"
    description = "Shortest total job first — best mean latency, can starve long jobs."

    def order(self, waiting, now_ms):
        return sorted(waiting, key=lambda r: (r.total_len, r.arrival_ms, r.id))


class PriorityPolicy(Policy):
    name = "priority"
    label = "Priority"
    description = "Higher priority first (e.g. interactive before batch)."

    def order(self, waiting, now_ms):
        return sorted(waiting, key=lambda r: (-r.priority, r.arrival_ms, r.id))

    def preempt_candidate(self, running, newcomer):
        candidates = [r for r in running if r.priority < newcomer.priority]
        if not candidates:
            return None
        return max(candidates, key=lambda r: (r.total_len - r.generated, r.id))


class AgingSJfPolicy(Policy):
    """SJF with aging: long jobs accumulate credit and eventually get served."""

    name = "sjf_aging"
    label = "SJF + aging"
    description = "SJF plus anti-starvation aging — keeps mean latency low without starving."

    def __init__(self, aging_tau_ms: float = 2000.0):
        self.tau = max(aging_tau_ms, 1.0)

    def _score(self, request: SimRequest, now_ms: float) -> float:
        waited = max(now_ms - request.arrival_ms, 0.0)
        return request.total_len / (1.0 + waited / self.tau)

    def order(self, waiting, now_ms):
        return sorted(waiting, key=lambda r: (self._score(r, now_ms), r.arrival_ms, r.id))


class RoundRobinPolicy(Policy):
    """FIFO, but requests are admitted in arrival slots — baseline for batching."""

    name = "round_robin"
    label = "Round robin"
    description = "Cyclic admission across arrival waves."

    def order(self, waiting, now_ms):
        return sorted(waiting, key=lambda r: (r.arrival_ms // 100.0, r.id))


POLICIES: dict[str, type[Policy]] = {
    FCFSPolicy.name: FCFSPolicy,
    SJfPolicy.name: SJfPolicy,
    PriorityPolicy.name: PriorityPolicy,
    AgingSJfPolicy.name: AgingSJfPolicy,
    RoundRobinPolicy.name: RoundRobinPolicy,
}


def get_policy(name: str, **kwargs) -> Policy:
    if name not in POLICIES:
        raise KeyError(f"Unknown policy {name!r}; available: {sorted(POLICIES)}")
    return POLICIES[name](**kwargs)
