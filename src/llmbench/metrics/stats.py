"""Small, dependency-free statistics helpers.

Implemented in pure Python so they can be unit-tested without numpy and
used inside the scheduler simulator (which must stay lightweight).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

__all__ = [
    "percentile",
    "summarize",
    "safe_div",
    "mean",
    "stdev",
    "coefficient_of_variation",
    "cdf_points",
    "histogram",
]


def _clean(values: Iterable[float | None]) -> list[float]:
    out: list[float] = []
    for v in values:
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isnan(f) or math.isinf(f):
            continue
        out.append(f)
    return out


def percentile(values: Sequence[float], q: float, method: str = "linear") -> float | None:
    """Percentile with numpy-compatible linear interpolation.

    Parameters
    ----------
    values: sequence of numbers (``None``/NaN entries are dropped)
    q: percentile in ``[0, 100]``
    method: ``"linear"`` (default), ``"nearest"`` or ``"lower"``
    """
    data = sorted(_clean(values))
    if not data:
        return None
    if len(data) == 1:
        return data[0]
    if not 0.0 <= q <= 100.0:
        raise ValueError(f"percentile must be in [0, 100], got {q}")

    pos = (len(data) - 1) * (q / 100.0)
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))

    if method == "nearest":
        return data[int(round(pos))]
    if method == "lower":
        return data[lower]
    if lower == upper:
        return data[lower]
    weight = pos - lower
    return data[lower] * (1.0 - weight) + data[upper] * weight


def mean(values: Sequence[float]) -> float | None:
    data = _clean(values)
    return sum(data) / len(data) if data else None


def stdev(values: Sequence[float], ddof: int = 1) -> float | None:
    data = _clean(values)
    if len(data) - ddof <= 0:
        return 0.0 if data else None
    mu = sum(data) / len(data)
    var = sum((x - mu) ** 2 for x in data) / (len(data) - ddof)
    return math.sqrt(var)


def summarize(values: Iterable[float | None]) -> dict[str, float | None]:
    """Return a compact statistical summary (``None`` for empty input)."""
    data = _clean(values)
    if not data:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": len(data),
        "mean": sum(data) / len(data),
        "std": stdev(data),
        "min": min(data),
        "p50": percentile(data, 50),
        "p90": percentile(data, 90),
        "p95": percentile(data, 95),
        "p99": percentile(data, 99),
        "max": max(data),
    }


def safe_div(numerator: float | None, denominator: float | None, default: float = 0.0) -> float:
    """Division that returns ``default`` when the denominator is 0/None."""
    if numerator is None or denominator in (None, 0):
        return default
    try:
        result = float(numerator) / float(denominator)
    except (TypeError, ValueError, ZeroDivisionError):
        return default
    if math.isnan(result) or math.isinf(result):
        return default
    return result


def coefficient_of_variation(values: Sequence[float]) -> float | None:
    """std / mean — useful to flag noisy measurements."""
    mu = mean(values)
    sd = stdev(values)
    if mu in (None, 0) or sd is None:
        return None
    return abs(sd / mu)


def cdf_points(values: Sequence[float]) -> tuple[list[float], list[float]]:
    """Return ``(sorted_values, cumulative_fraction)`` for a CDF plot."""
    data = sorted(_clean(values))
    if not data:
        return [], []
    n = len(data)
    return data, [(i + 1) / n for i in range(n)]


def histogram(values: Sequence[float], bins: int = 20) -> tuple[list[float], list[int]]:
    """Simple equal-width histogram returning ``(bin_edges, counts)``."""
    data = _clean(values)
    if not data:
        return [], []
    lo, hi = min(data), max(data)
    if hi == lo:
        return [lo, lo + 1e-9], [len(data)]
    width = (hi - lo) / bins
    edges = [lo + i * width for i in range(bins + 1)]
    counts = [0] * bins
    for v in data:
        idx = min(int((v - lo) / width), bins - 1)
        counts[idx] += 1
    return edges, counts
