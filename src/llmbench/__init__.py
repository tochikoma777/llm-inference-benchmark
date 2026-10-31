"""llmbench — reproducible benchmarking harness for LLM serving backends.

The package is organised in four layers:

``core``       configuration, logging, paths, result storage
``backends``   inference engines (vLLM, HuggingFace, deterministic mock)
``runner``     workload generation, load generation, benchmark scenarios
``viz``        plotting theme, static figures, interactive HTML/dashboard

Heavy GPU dependencies (torch/vllm) are imported lazily inside the backend
modules so that analysis and reporting keep working on CPU-only machines.
"""

from __future__ import annotations

__version__ = "0.2.1"
__all__ = ["__version__"]
