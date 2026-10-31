"""Discrete-event scheduler simulator (no GPU required).

Models the decisions a serving engine makes every iteration: which requests to
admit, how to batch them, when KV-cache blocks run out, and how preemption
affects tail latency. Used to compare FCFS / SJF / priority / aging policies on
an identical workload.
"""

from __future__ import annotations

from .engine import PolicyResult, Simulation
from .model import BlockPool, SimRequest, generate_workload
from .policies import POLICIES, Policy, get_policy

__all__ = [
    "PolicyResult",
    "Simulation",
    "BlockPool",
    "SimRequest",
    "generate_workload",
    "POLICIES",
    "Policy",
    "get_policy",
]
