"""Statistics helpers."""

from __future__ import annotations

import math

import pytest

from llmbench.metrics.stats import (
    cdf_points,
    coefficient_of_variation,
    histogram,
    mean,
    percentile,
    safe_div,
    stdev,
    summarize,
)


def test_percentile_matches_numpy_linear_interpolation():
    values = [1, 2, 3, 4]
    assert percentile(values, 0) == 1
    assert percentile(values, 100) == 4
    assert percentile(values, 50) == pytest.approx(2.5)
    assert percentile(values, 25) == pytest.approx(1.75)


def test_percentile_handles_small_and_empty_inputs():
    assert percentile([], 95) is None
    assert percentile([7], 95) == 7


def test_percentile_methods():
    values = [10, 20, 30, 40]
    assert percentile(values, 90, method="nearest") in values
    assert percentile(values, 90, method="lower") == 30


def test_percentile_rejects_invalid_quantile():
    with pytest.raises(ValueError):
        percentile([1, 2, 3], 101)


def test_percentile_drops_nan_and_none():
    assert percentile([1, None, 3, float("nan")], 50) == 2


def test_summarize_returns_expected_keys():
    summary = summarize([5, 1, 3])
    assert summary["count"] == 3
    assert summary["min"] == 1
    assert summary["max"] == 5
    assert summary["mean"] == pytest.approx(3)
    assert summary["p50"] == 3


def test_summarize_empty_is_all_none():
    summary = summarize([])
    assert summary["count"] == 0
    assert summary["mean"] is None


def test_stdev_sample_versus_population():
    values = [2, 4, 4, 4, 5, 5, 7, 9]
    assert stdev(values, ddof=0) == pytest.approx(2.0)
    assert stdev(values, ddof=1) == pytest.approx(2.138, rel=1e-3)
    assert stdev([1.0]) == 0.0


def test_safe_div():
    assert safe_div(10, 2) == 5
    assert safe_div(10, 0) == 0
    assert safe_div(None, 5) == 0
    assert safe_div(float("inf"), 1.0) == 0


def test_mean_and_cv():
    assert mean([]) is None
    assert mean([1, 2, 3]) == 2
    assert coefficient_of_variation([1, 1, 1]) == 0


def test_cdf_points_and_histogram():
    xs, ys = cdf_points([3, 1, 2])
    assert xs == [1, 2, 3]
    assert ys == pytest.approx([1 / 3, 2 / 3, 1.0])

    edges, counts = histogram([1, 1, 2, 5], bins=2)
    assert len(edges) == 3
    assert sum(counts) == 4


def test_histogram_constant_values():
    edges, counts = histogram([4, 4, 4], bins=5)
    assert counts == [3]
    assert not math.isnan(edges[0])
