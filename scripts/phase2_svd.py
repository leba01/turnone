"""Phase 2 Experiment 1: SVD / Effective Rank analysis.

Reads cached payoff matrices and computes effective rank, spectral structure,
and payoff-space projections. Tests whether the high TV(BC, Nash) = 0.99
is inflated by payoff-irrelevant dimensions.

Usage:
    python scripts/phase2_svd.py \
        --cache results/phase2/cache.pkl \
        --out results/phase2/svd.json
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from turnone.eval.bootstrap import bootstrap_all


def _effective_rank(S: np.ndarray, threshold: float) -> int:
    """Minimum k such that sum(S[:k]^2) / sum(S^2) >= threshold."""
    energy = np.cumsum(S ** 2)
    total = energy[-1]
    if total < 1e-15:
        return 1
    fracs = energy / total
    k = int(np.searchsorted(fracs, threshold)) + 1
    return min(k, len(S))


def _top_k_variance_fraction(S: np.ndarray, k: int) -> float:
    """Fraction of Frobenius norm in top-k singular values."""
    total = np.sum(S ** 2)
    if total < 1e-15:
        return 1.0
    return float(np.sum(S[:k] ** 2) / total)



def analyze_matchup(m: dict) -> dict:
    """SVD analysis for a single matchup."""
    R = np.asarray(m["R"], dtype=np.float64)
    bc_p1 = np.asarray(m["bc_p1"], dtype=np.float64)
    bc_p2 = np.asarray(m["bc_p2"], dtype=np.float64)
    nash_p1 = np.asarray(m["nash_p1"], dtype=np.float64)
    nash_p2 = np.asarray(m["nash_p2"], dtype=np.float64)

    n1, n2 = R.shape

    # Full SVD
    U, S, Vt = np.linalg.svd(R, full_matrices=False)

    # Effective rank at various thresholds
    eff_rank = {
        f"eff_rank_{int(t*100)}": _effective_rank(S, t)
        for t in [0.90, 0.95, 0.99]
    }

    # Top-k variance fractions
    top_k_frac = {
        f"top_{k}_frac": _top_k_variance_fraction(S, k)
        for k in [1, 2, 3, 5, 10]
    }

    # Spectral gap
    spectral_gap = float(S[0] / S[1]) if len(S) > 1 and S[1] > 1e-15 else float("inf")

    # Nash support sizes
    nash_support_p1 = int((nash_p1 > 1e-4).sum())
    nash_support_p2 = int((nash_p2 > 1e-4).sum())

    # Original TV for reference
    tv_orig_p1 = 0.5 * float(np.abs(bc_p1 - nash_p1).sum())
    tv_orig_p2 = 0.5 * float(np.abs(bc_p2 - nash_p2).sum())

    # Payoff-weighted TV: weight strategy differences by action payoff relevance
    # Uses S-weighted projection to measure "payoff-relevant distance"
    # p1_payoff_relevance[i] = || U[i,:] * S || — how much action i participates in payoff
    payoff_weight_p1 = np.sqrt(np.sum((U * S[None, :]) ** 2, axis=1))  # (n1,)
    payoff_weight_p2 = np.sqrt(np.sum((Vt.T * S[None, :]) ** 2, axis=1))  # (n2,)

    # Payoff-weighted TV: sum |bc - nash| * weight / sum(weight)
    pw_p1 = payoff_weight_p1 / payoff_weight_p1.sum()
    pw_p2 = payoff_weight_p2 / payoff_weight_p2.sum()
    payoff_wtd_tv_p1 = float(np.sum(np.abs(bc_p1 - nash_p1) * pw_p1))
    payoff_wtd_tv_p2 = float(np.sum(np.abs(bc_p2 - nash_p2) * pw_p2))

    # BC exploitability (for correlation analysis)
    game_value = m["game_value"]
    col_payoffs = bc_p1 @ R
    bc_exploit = game_value - float(col_payoffs.min())

    return {
        "idx": m["idx"],
        "n1": n1,
        "n2": n2,
        "min_dim": min(n1, n2),
        **eff_rank,
        **top_k_frac,
        "spectral_gap": spectral_gap,
        "nash_support_p1": nash_support_p1,
        "nash_support_p2": nash_support_p2,
        "tv_orig_p1": tv_orig_p1,
        "tv_orig_p2": tv_orig_p2,
        "payoff_wtd_tv_p1": payoff_wtd_tv_p1,
        "payoff_wtd_tv_p2": payoff_wtd_tv_p2,
        "bc_exploitability": bc_exploit,
        "game_value": game_value,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 2 Exp 1: SVD effective rank")
    parser.add_argument("--cache", required=True, help="Path to cache.pkl")
    parser.add_argument("--out", required=True, help="Output JSON path")
    args = parser.parse_args()

    with open(args.cache, "rb") as f:
        matchups = pickle.load(f)
    print(f"Loaded {len(matchups)} matchups from cache")

    # Analyze each matchup
    results = [analyze_matchup(m) for m in matchups]

    # Aggregate
    n = len(results)

    # Effective rank stats
    for threshold in [90, 95, 99]:
        key = f"eff_rank_{threshold}"
        vals = np.array([r[key] for r in results], dtype=float)
        boot = bootstrap_all({key: vals}, n_resamples=10_000, seed=42)
        print(f"Effective rank ({threshold}%): "
              f"mean {boot[key]['mean']:.1f} [{boot[key]['mean_ci_lo']:.1f}, {boot[key]['mean_ci_hi']:.1f}], "
              f"median {boot[key]['median']:.1f}")

    # Top-k fractions
    for k in [1, 2, 3, 5, 10]:
        key = f"top_{k}_frac"
        vals = np.array([r[key] for r in results])
        print(f"Top-{k} variance fraction: mean {vals.mean():.3f}, median {np.median(vals):.3f}")

    # Spectral gap
    gaps = np.array([r["spectral_gap"] for r in results])
    finite_gaps = gaps[np.isfinite(gaps)]
    print(f"Spectral gap (S[0]/S[1]): mean {finite_gaps.mean():.2f}, "
          f"median {np.median(finite_gaps):.2f}")

    # Correlations
    eff_ranks_95 = np.array([r["eff_rank_95"] for r in results], dtype=float)
    exploits = np.array([r["bc_exploitability"] for r in results])
    game_sizes = np.array([r["n1"] * r["n2"] for r in results], dtype=float)
    nash_supports = np.array([r["nash_support_p1"] for r in results], dtype=float)

    print(f"\nCorrelations with effective rank (95%):")
    print(f"  vs exploitability: {np.corrcoef(eff_ranks_95, exploits)[0,1]:.3f}")
    print(f"  vs game size (n1*n2): {np.corrcoef(eff_ranks_95, game_sizes)[0,1]:.3f}")
    print(f"  vs Nash support P1: {np.corrcoef(eff_ranks_95, nash_supports)[0,1]:.3f}")

    # Strategy distances
    tv_orig_p1 = np.array([r["tv_orig_p1"] for r in results])
    tv_orig_p2 = np.array([r["tv_orig_p2"] for r in results])
    pw_tv_p1 = np.array([r["payoff_wtd_tv_p1"] for r in results])
    pw_tv_p2 = np.array([r["payoff_wtd_tv_p2"] for r in results])
    print(f"\nStrategy distance (BC vs Nash):")
    print(f"  Original TV:        P1 mean {tv_orig_p1.mean():.3f}, P2 mean {tv_orig_p2.mean():.3f}")
    print(f"  Payoff-weighted TV: P1 mean {pw_tv_p1.mean():.3f}, P2 mean {pw_tv_p2.mean():.3f}")

    # Build bootstrap CIs for all key metrics
    boot_data = {
        "eff_rank_90": np.array([r["eff_rank_90"] for r in results], dtype=float),
        "eff_rank_95": eff_ranks_95,
        "eff_rank_99": np.array([r["eff_rank_99"] for r in results], dtype=float),
        "top_1_frac": np.array([r["top_1_frac"] for r in results]),
        "top_3_frac": np.array([r["top_3_frac"] for r in results]),
        "spectral_gap": finite_gaps,
    }
    boot_results = bootstrap_all(boot_data, n_resamples=10_000, seed=42)

    # Save
    output = {
        "n_matchups": n,
        "per_matchup": results,
        "aggregate": {
            "bootstrap": boot_results,
            "correlations": {
                "eff_rank_95_vs_exploitability": float(np.corrcoef(eff_ranks_95, exploits)[0, 1]),
                "eff_rank_95_vs_game_size": float(np.corrcoef(eff_ranks_95, game_sizes)[0, 1]),
                "eff_rank_95_vs_nash_support": float(np.corrcoef(eff_ranks_95, nash_supports)[0, 1]),
            },
            "top_k_frac_means": {
                f"top_{k}": float(np.mean([r[f"top_{k}_frac"] for r in results]))
                for k in [1, 2, 3, 5, 10]
            },
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
