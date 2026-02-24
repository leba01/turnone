"""Tests for bootstrap CI utilities."""

from __future__ import annotations

import numpy as np
import pytest

from turnone.eval.bootstrap import bootstrap_ci, bootstrap_all


class TestBootstrapCI:
    """Tests for bootstrap_ci."""

    def test_basic_mean(self):
        """Mean CI should bracket the true mean for a simple case."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pt, lo, hi = bootstrap_ci(values, n_resamples=5000, ci=0.95, statistic="mean")
        assert pt == pytest.approx(3.0)
        assert lo <= 3.0
        assert hi >= 3.0

    def test_basic_median(self):
        """Median CI should bracket the true median."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pt, lo, hi = bootstrap_ci(values, n_resamples=5000, ci=0.95, statistic="median")
        assert pt == pytest.approx(3.0)
        assert lo <= pt <= hi

    def test_narrow_ci_for_constant(self):
        """Constant values should give very narrow CI."""
        values = np.full(100, 5.0)
        pt, lo, hi = bootstrap_ci(values, n_resamples=1000, ci=0.95, statistic="mean")
        assert pt == pytest.approx(5.0)
        assert abs(hi - lo) < 1e-10

    def test_wider_ci_for_noisy_data(self):
        """Higher variance data should give wider CI."""
        rng = np.random.RandomState(42)
        narrow = rng.randn(100) * 0.1
        wide = rng.randn(100) * 10.0

        _, lo_n, hi_n = bootstrap_ci(narrow, n_resamples=5000, statistic="mean", seed=0)
        _, lo_w, hi_w = bootstrap_ci(wide, n_resamples=5000, statistic="mean", seed=0)

        assert (hi_w - lo_w) > (hi_n - lo_n)

    def test_ci_ordering(self):
        """CI lower should be <= point estimate <= CI upper."""
        values = np.random.RandomState(42).randn(50)
        pt, lo, hi = bootstrap_ci(values, n_resamples=5000, ci=0.95, statistic="mean")
        assert lo <= pt <= hi


class TestBootstrapAll:
    """Tests for bootstrap_all."""

    def test_output_structure(self):
        """Should return correct keys for each metric."""
        data = {
            "metric_a": np.random.RandomState(42).randn(50),
            "metric_b": np.random.RandomState(43).randn(50),
        }
        results = bootstrap_all(data, n_resamples=1000)

        assert set(results.keys()) == {"metric_a", "metric_b"}
        for name in results:
            assert "mean" in results[name]
            assert "mean_ci_lo" in results[name]
            assert "mean_ci_hi" in results[name]
            assert "median" in results[name]
            assert "median_ci_lo" in results[name]
            assert "median_ci_hi" in results[name]

    def test_empty_array(self):
        """Empty arrays should return zeros."""
        data = {"empty": np.array([])}
        results = bootstrap_all(data)
        assert results["empty"]["mean"] == 0.0
