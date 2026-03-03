"""Counterfactual strategy gap analysis.

Tests whether the small bc_vs_bc - V* gap is specific to expert strategies
or a structural property of low-rank games. Generates random, shuffled,
and uniform strategies and measures their cross-play gaps against V*.

Key insight: if the game is truly low-rank (eff. rank 3), then ARBITRARY
strategy pairs (p, q) should satisfy |p^T R q - V*| ≈ 0, because most
dimensions of any strategy are payoff-irrelevant.

Usage:
    python scripts/counterfactual_analysis.py \
        --cache results/phase2/cache.pkl \
        --out results/counterfactual/counterfactual.json
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from turnone.eval.bootstrap import bootstrap_all


def random_simplex(n: int, rng: np.random.RandomState) -> np.ndarray:
    """Sample a random point on the n-simplex (Dirichlet(1,...,1))."""
    x = rng.exponential(1.0, size=n)
    return x / x.sum()


def shuffled_strategy(original: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Permute the entries of a strategy vector."""
    perm = rng.permutation(len(original))
    return original[perm]


def analyze_matchup(m: dict, rng: np.random.RandomState, n_random: int = 50) -> dict:
    """Compute cross-play gaps for various counterfactual strategy types."""
    R = np.asarray(m["R"], dtype=np.float64)
    bc_p1 = np.asarray(m["bc_p1"], dtype=np.float64)
    bc_p2 = np.asarray(m["bc_p2"], dtype=np.float64)
    game_value = m["game_value"]
    n1, n2 = R.shape

    # BC gap (baseline)
    bc_gap = float(bc_p1 @ R @ bc_p2) - game_value

    # Uniform strategies
    unif_p1 = np.ones(n1) / n1
    unif_p2 = np.ones(n2) / n2
    uniform_gap = float(unif_p1 @ R @ unif_p2) - game_value

    # Shuffled BC strategies (permute action labels)
    shuffled_gaps = []
    for _ in range(n_random):
        sp1 = shuffled_strategy(bc_p1, rng)
        sp2 = shuffled_strategy(bc_p2, rng)
        shuffled_gaps.append(float(sp1 @ R @ sp2) - game_value)

    # Random simplex strategies
    random_gaps = []
    for _ in range(n_random):
        rp1 = random_simplex(n1, rng)
        rp2 = random_simplex(n2, rng)
        random_gaps.append(float(rp1 @ R @ rp2) - game_value)

    return {
        "idx": m["idx"],
        "game_value": game_value,
        "n_actions_p1": n1,
        "n_actions_p2": n2,
        "bc_gap": bc_gap,
        "uniform_gap": uniform_gap,
        "shuffled_gap_mean": float(np.mean(shuffled_gaps)),
        "shuffled_gap_std": float(np.std(shuffled_gaps)),
        "shuffled_gap_abs_mean": float(np.mean(np.abs(shuffled_gaps))),
        "random_gap_mean": float(np.mean(random_gaps)),
        "random_gap_std": float(np.std(random_gaps)),
        "random_gap_abs_mean": float(np.mean(np.abs(random_gaps))),
    }


def main():
    parser = argparse.ArgumentParser(description="Counterfactual strategy gap analysis")
    parser.add_argument("--cache", required=True, help="Path to cache.pkl")
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument("--n_random", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.cache, "rb") as f:
        matchups = pickle.load(f)
    print(f"Loaded {len(matchups)} matchups from cache")

    rng = np.random.RandomState(args.seed)
    results = [analyze_matchup(m, rng, args.n_random) for m in matchups]
    n = len(results)

    # Aggregate
    bc_gaps = np.array([r["bc_gap"] for r in results])
    uniform_gaps = np.array([r["uniform_gap"] for r in results])
    shuffled_gaps = np.array([r["shuffled_gap_mean"] for r in results])
    random_gaps = np.array([r["random_gap_mean"] for r in results])
    shuffled_abs = np.array([r["shuffled_gap_abs_mean"] for r in results])
    random_abs = np.array([r["random_gap_abs_mean"] for r in results])

    print(f"\n{'=' * 60}")
    print("COUNTERFACTUAL STRATEGY GAP ANALYSIS")
    print(f"{'=' * 60}")
    print(f"\nStrategy type           | Mean gap  | Mean |gap|  | Std")
    print(f"-" * 60)
    print(
        f"BC (expert)             | {np.mean(bc_gaps):+.4f}  | {np.mean(np.abs(bc_gaps)):.4f}    | {np.std(bc_gaps):.4f}"
    )
    print(
        f"Uniform                 | {np.mean(uniform_gaps):+.4f}  | {np.mean(np.abs(uniform_gaps)):.4f}    | {np.std(uniform_gaps):.4f}"
    )
    print(
        f"Shuffled BC             | {np.mean(shuffled_gaps):+.4f}  | {np.mean(shuffled_abs):.4f}    | {np.std(shuffled_gaps):.4f}"
    )
    print(
        f"Random (Dirichlet)      | {np.mean(random_gaps):+.4f}  | {np.mean(random_abs):.4f}    | {np.std(random_gaps):.4f}"
    )

    # Bootstrap
    boot_data = {
        "bc_gap": bc_gaps,
        "uniform_gap": uniform_gaps,
        "shuffled_gap": shuffled_gaps,
        "random_gap": random_gaps,
        "bc_abs_gap": np.abs(bc_gaps),
        "uniform_abs_gap": np.abs(uniform_gaps),
        "shuffled_abs_gap": shuffled_abs,
        "random_abs_gap": random_abs,
    }
    boot = bootstrap_all(boot_data, n_resamples=10_000, seed=42)

    print(f"\nBootstrap 95% CIs (mean gap):")
    for key in ["bc_gap", "uniform_gap", "shuffled_gap", "random_gap"]:
        b = boot[key]
        print(
            f"  {key}: {b['mean']:.4f} [{b['mean_ci_lo']:.4f}, {b['mean_ci_hi']:.4f}]"
        )

    print(f"\nBootstrap 95% CIs (mean |gap|):")
    for key in ["bc_abs_gap", "uniform_abs_gap", "shuffled_abs_gap", "random_abs_gap"]:
        b = boot[key]
        print(
            f"  {key}: {b['mean']:.4f} [{b['mean_ci_lo']:.4f}, {b['mean_ci_hi']:.4f}]"
        )

    # Formal price-of-convention bound (theoretical)
    # For a rank-k game R ≈ R_k, with ||R - R_k||_F / ||R||_F = ε,
    # the gap |p^T R q - p^T R_k q| ≤ ||p|| ||R - R_k||_F ||q|| = ε ||R||_F
    # (since ||p||=||q||=1 on simplex, actually ||p||≤1)
    # More precisely: p^T (R - R_k) q ≤ σ_{k+1} (by spectral norm bound)
    print(f"\n{'=' * 60}")
    print("PRICE OF CONVENTION BOUND (THEORETICAL)")
    print(f"{'=' * 60}")
    print("For rank-k approximation R_k with residual energy ε:")
    print("  |p^T R q - p^T R_k q| ≤ σ_{k+1}  (spectral norm bound)")
    print("  |V*(R) - V*(R_k)| ≤ σ_{k+1}      (von Neumann minimax)")
    print("  Combined: |p^T R q - V*(R)| ≤ |p^T R_k q - V*(R_k)| + 2σ_{k+1}")
    print("  In a rank-k game, V*(R_k) = V*(R) and p^T R_k q = V* for generic p,q")
    print("  So the price of convention ≤ 2σ_{k+1} ≈ 2 × residual")

    # Save
    output = {
        "n_matchups": n,
        "n_random_per_matchup": args.n_random,
        "seed": args.seed,
        "per_matchup": results,
        "aggregate": {
            "summary": {
                "bc_gap_mean": float(np.mean(bc_gaps)),
                "bc_gap_abs_mean": float(np.mean(np.abs(bc_gaps))),
                "uniform_gap_mean": float(np.mean(uniform_gaps)),
                "uniform_gap_abs_mean": float(np.mean(np.abs(uniform_gaps))),
                "shuffled_gap_mean": float(np.mean(shuffled_gaps)),
                "shuffled_gap_abs_mean": float(np.mean(shuffled_abs)),
                "random_gap_mean": float(np.mean(random_gaps)),
                "random_gap_abs_mean": float(np.mean(random_abs)),
            },
            "bootstrap": boot,
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
