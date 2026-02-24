"""Bootstrap confidence interval utilities."""

from __future__ import annotations

import numpy as np


def bootstrap_ci(
    values: np.ndarray,
    n_resamples: int = 10_000,
    ci: float = 0.95,
    statistic: str = "mean",
    seed: int = 0,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for a statistic.

    Args:
        values: (N,) array of per-matchup values.
        n_resamples: number of bootstrap resamples.
        ci: confidence level (e.g. 0.95 for 95% CI).
        statistic: "mean" or "median".
        seed: random seed.

    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    rng = np.random.RandomState(seed)
    n = len(values)

    stat_fn = np.mean if statistic == "mean" else np.median
    point = float(stat_fn(values))

    boot_stats = np.empty(n_resamples)
    for i in range(n_resamples):
        sample = values[rng.randint(0, n, size=n)]
        boot_stats[i] = stat_fn(sample)

    alpha = (1.0 - ci) / 2.0
    ci_lower = float(np.percentile(boot_stats, 100 * alpha))
    ci_upper = float(np.percentile(boot_stats, 100 * (1 - alpha)))

    return point, ci_lower, ci_upper


def bootstrap_all(
    data: dict[str, np.ndarray],
    n_resamples: int = 10_000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Compute bootstrap CIs for multiple named arrays.

    Args:
        data: dict mapping metric name -> (N,) array of per-matchup values.
        n_resamples: number of bootstrap resamples.
        ci: confidence level.
        seed: random seed.

    Returns:
        Dict mapping metric name -> {mean, mean_ci_lo, mean_ci_hi,
                                      median, median_ci_lo, median_ci_hi}.
    """
    results = {}
    for name, values in data.items():
        if len(values) == 0:
            results[name] = {
                "mean": 0.0, "mean_ci_lo": 0.0, "mean_ci_hi": 0.0,
                "median": 0.0, "median_ci_lo": 0.0, "median_ci_hi": 0.0,
            }
            continue

        mean_pt, mean_lo, mean_hi = bootstrap_ci(
            values, n_resamples, ci, "mean", seed)
        med_pt, med_lo, med_hi = bootstrap_ci(
            values, n_resamples, ci, "median", seed)

        results[name] = {
            "mean": mean_pt, "mean_ci_lo": mean_lo, "mean_ci_hi": mean_hi,
            "median": med_pt, "median_ci_lo": med_lo, "median_ci_hi": med_hi,
        }

    return results
