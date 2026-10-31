"""Prompt datasets and workload construction.

The harness can consume either a JSONL dataset (one object per line with a
``prompt`` field) or a procedurally generated mixture. Generating prompts at
runtime means a fresh clone has something to benchmark immediately, without
needing a tokenizer or a 2 GB download.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from ..core.logging import get_logger
from ..core.paths import prompts_dir

__all__ = [
    "PromptSample",
    "estimate_tokens",
    "load_prompt_file",
    "synthesize_prompt",
    "build_workload",
    "write_synthetic_dataset",
]

log = get_logger("runner.workload")

#: Characters-per-token heuristic. Qwen-style tokenizers average ~1.7–1.9
#: characters per token on mixed Chinese/English prose.
CHARS_PER_TOKEN = 1.8

TOPICS: Sequence[str] = (
    "大语言模型基于 Transformer 架构，通过自注意力机制处理序列数据。预训练阶段使用海量文本，"
    "通过预测下一个 token 学习语言表示；微调阶段使用特定领域数据对齐模型行为。",
    "推理阶段的核心瓶颈包括内存带宽受限、KV Cache 管理与批量调度。PagedAttention 借鉴操作系统的"
    "虚拟内存分页思想，把 KV Cache 切分成固定大小的 block 按需分配，显著降低显存碎片。",
    "连续批处理（continuous batching）允许在序列级别动态插入新请求，避免静态批处理中"
    "短请求必须等待长请求完成的问题，从而把 GPU 利用率从不足 20% 提升到 80% 以上。",
    "前缀缓存（prefix caching）把多个请求共享的 system prompt 或文档前缀的 KV 计算复用起来，"
    "在 RAG 与多轮对话场景中显著降低首 token 延迟。",
    "量化通过降低权重精度来减少显存占用与带宽需求，常见方案包括 FP8、INT8 与 4-bit 权重量化。"
    "权重量化主要影响 prefill 的算术强度，KV Cache 量化则直接决定可容纳的并发序列数。",
    "调度策略决定公平性与效率的权衡：FCFS 保证先来先服务，而 SJF 能降低平均延迟，"
    "却可能让长请求饥饿。生产中通常采用带优先级的公平调度并配合抢占。",
    "GPU 显存带宽是解码阶段的主要约束。以 7B 模型 FP16 为例，每生成一个 token 需要读取约 14GB "
    "权重，因此在 HBM 带宽受限时提高 batch size 是提升吞吐最直接的手段。",
    "可观测性是推理服务的基石：TTFT、TPOT、P99 延迟、队列等待时间与显存水位需要持续采集，"
    "才能在流量突增时定位瓶颈究竟是计算、显存还是调度。",
)

PROMPT_SUFFIXES: Sequence[str] = (
    "\n\n请用三点总结上述内容，并说明各自的关键取舍。",
    "\n\n请解释其中的核心机制，并给出一个真实场景的例子。",
    "\n\n请分析这些技术之间的相互关系，并指出最常见的误区。",
    "\n\n请分别说明在低并发与高并发场景下的最优选择。",
)


@dataclass
class PromptSample:
    """A single prompt plus its intended token length."""

    text: str
    target_tokens: int
    actual_tokens: int | None = None
    source: str = "synthetic"

    @property
    def prompt(self) -> str:
        """Alias kept for readability in scenario code."""
        return self.text


def estimate_tokens(text: str, chars_per_token: float = CHARS_PER_TOKEN) -> int:
    """Cheap token estimate for tokenizers we cannot load."""
    if not text:
        return 0
    return max(1, int(len(text) / chars_per_token))


def synthesize_prompt(
    target_tokens: int,
    rng: random.Random | None = None,
    with_instruction: bool = True,
) -> str:
    """Build a prompt whose estimated token count is close to ``target_tokens``."""
    rng = rng or random.Random(0)
    budget_chars = max(int(target_tokens * CHARS_PER_TOKEN), 32)
    parts: list[str] = []
    used = 0
    while used < budget_chars:
        topic = rng.choice(TOPICS)
        parts.append(topic + "\n\n")
        used += len(topic) + 2
    text = "".join(parts)[:budget_chars]
    if with_instruction:
        text += rng.choice(PROMPT_SUFFIXES)
    return text


def load_prompt_file(path: Path) -> list[PromptSample]:
    """Read a JSONL prompt file (``{"prompt": ..., "actual_length": ...}``)."""
    samples: list[PromptSample] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                log.warning("Skipping malformed line %d in %s", line_no, path)
                continue
            text = row.get("prompt") or row.get("text") or ""
            if not text:
                continue
            actual = row.get("actual_length") or row.get("tokens")
            samples.append(
                PromptSample(
                    text=text,
                    target_tokens=int(row.get("target_length") or actual or estimate_tokens(text)),
                    actual_tokens=int(actual) if actual else None,
                    source=str(path.name),
                )
            )
    if not samples:
        raise ValueError(f"No usable prompts found in {path}")
    return samples


def _default_mix(rng: random.Random, n: int) -> list[PromptSample]:
    """60 % short / 40 % long mixture, mirroring a real chat+document workload."""
    samples: list[PromptSample] = []
    for _i in range(n):
        target = rng.randint(96, 320) if rng.random() < 0.6 else rng.randint(512, 1536)
        samples.append(PromptSample(text=synthesize_prompt(target, rng), target_tokens=target))
    return samples


def build_workload(
    n: int,
    dataset: str | None = None,
    seed: int = 42,
    lengths: Iterable[int] | None = None,
) -> list[PromptSample]:
    """Return exactly ``n`` prompts, cycling the dataset when needed.

    Parameters
    ----------
    n: number of prompts required
    dataset: path to a JSONL dataset; when ``None`` a synthetic mixture is used
    seed: RNG seed for reproducibility
    lengths: optional explicit target lengths (cycled)
    """
    rng = random.Random(seed)

    if dataset:
        path = Path(dataset).expanduser()
        if not path.is_absolute() and not path.exists():
            path = prompts_dir() / path
        if not path.exists():
            raise FileNotFoundError(f"Prompt dataset not found: {dataset}")
        base = load_prompt_file(path)
    elif lengths:
        wanted = list(lengths)
        base = [
            PromptSample(
                text=synthesize_prompt(wanted[i % len(wanted)], rng),
                target_tokens=wanted[i % len(wanted)],
            )
            for i in range(max(n, len(wanted)))
        ]
    else:
        base = _default_mix(rng, max(n, 8))

    if not base:
        raise ValueError("Workload is empty")

    out: list[PromptSample] = []
    for i in range(n):
        sample = base[i % len(base)]
        out.append(
            PromptSample(
                text=sample.text,
                target_tokens=sample.target_tokens,
                actual_tokens=sample.actual_tokens,
                source=sample.source,
            )
        )
    return out


def write_synthetic_dataset(
    path: Path,
    num_prompts: int = 120,
    seed: int = 42,
    buckets: Sequence[tuple[str, int, int]] = (
        ("short", 128, 256),
        ("medium", 384, 640),
        ("long", 896, 1280),
        ("very_long", 1792, 2304),
    ),
) -> Path:
    """Write a balanced synthetic dataset (useful for `llmbench datasets`)."""
    rng = random.Random(seed)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    per_bucket = max(num_prompts // len(buckets), 1)
    for name, lo, hi in buckets:
        for _ in range(per_bucket):
            target = rng.randint(lo, hi)
            text = synthesize_prompt(target, rng)
            rows.append(
                {
                    "bucket": name,
                    "target_length": target,
                    "actual_length": estimate_tokens(text),
                    "prompt": text,
                }
            )
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    log.info("Wrote %d synthetic prompts to %s", len(rows), path)
    return path
