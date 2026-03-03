"""Projection analysis: where do BC-Nash deviations live relative to R's column/row space?

Tests the mechanistic claim: expert deviations are in the payoff null space.
For each matchup, projects δp = bc_p1 - nash_p1 onto R's column space (top-k
left singular vectors) and δq = bc_p2 - nash_p2 onto R's row space (top-k
right singular vectors).

If the low-rank explanation is correct, most deviation energy should be
ORTHOGONAL to the top SVs (fraction in top-k << 1).

Usage:
    python scripts/projection_analysis.py \
        --cache results/phase2/cache.pkl \
        --out results/projection/projection.json
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from turnone.eval.bootstrap import bootstrap_all


def analyze_matchup(m: dict) -> dict:
    """Project BC-Nash deviations onto R's singular subspaces."""
    R = np.asarray(m["R"], dtype=np.float64)
    bc_p1 = np.asarray(m["bc_p1"], dtype=np.float64)
    bc_p2 = np.asarray(m["bc_p2"], dtype=np.float64)
    nash_p1 = np.asarray(m["nash_p1"], dtype=np.float64)
    nash_p2 = np.asarray(m["nash_p2"], dtype=np.float64)

    U, S, Vt = np.linalg.svd(R, full_matrices=False)

    delta_p = bc_p1 - nash_p1  # deviation in P1's action space
    delta_q = bc_p2 - nash_p2  # deviation in P2's action space

    norm_dp = np.linalg.norm(delta_p)
    norm_dq = np.linalg.norm(delta_q)

    result = {
        "idx": m["idx"],
        "n1": R.shape[0],
        "n2": R.shape[1],
        "norm_delta_p": float(norm_dp),
        "norm_delta_q": float(norm_dq),
    }

    # Project onto top-k dimensions
    for k in [1, 2, 3, 5, 10]:
        if k > min(R.shape):
            continue

        # P1: project δp onto column space of R (spanned by U[:, :k])
        # ||U_k^T δp|| / ||δp||
        if norm_dp > 1e-15:
            proj_p = U[:, :k].T @ delta_p  # (k,)
            frac_p = float(np.linalg.norm(proj_p) / norm_dp)
        else:
            frac_p = 0.0

        # P2: project δq onto row space of R (spanned by Vt[:k, :].T)
        # ||V_k δq|| / ||δq||
        if norm_dq > 1e-15:
            proj_q = Vt[:k, :] @ delta_q  # (k,)
            frac_q = float(np.linalg.norm(proj_q) / norm_dq)
        else:
            frac_q = 0.0

        result[f"frac_p_top{k}"] = frac_p
        result[f"frac_q_top{k}"] = frac_q
        result[f"frac_avg_top{k}"] = (frac_p + frac_q) / 2

    # Also compute fraction of deviation energy (squared)
    for k in [1, 2, 3, 5, 10]:
        if k > min(R.shape):
            continue
        if norm_dp > 1e-15:
            proj_p = U[:, :k].T @ delta_p
            energy_p = float(np.sum(proj_p**2) / (norm_dp**2))
        else:
            energy_p = 0.0
        if norm_dq > 1e-15:
            proj_q = Vt[:k, :] @ delta_q
            energy_q = float(np.sum(proj_q**2) / (norm_dq**2))
        else:
            energy_q = 0.0
        result[f"energy_p_top{k}"] = energy_p
        result[f"energy_q_top{k}"] = energy_q
        result[f"energy_avg_top{k}"] = (energy_p + energy_q) / 2

    # Complementary: fraction in null space (orthogonal complement of top-3)
    if min(R.shape) >= 3:
        result["frac_p_null3"] = 1.0 - result.get("energy_p_top3", 0.0)
        result["frac_q_null3"] = 1.0 - result.get("energy_q_top3", 0.0)

    return result


def main():
    parser = argparse.ArgumentParser(description="Projection analysis")
    parser.add_argument("--cache", required=True, help="Path to cache.pkl")
    parser.add_argument("--out", required=True, help="Output JSON path")
    args = parser.parse_args()

    with open(args.cache, "rb") as f:
        matchups = pickle.load(f)
    print(f"Loaded {len(matchups)} matchups from cache")

    results = [analyze_matchup(m) for m in matchups]
    n = len(results)

    # Summary statistics
    print(f"\n{'=' * 60}")
    print("PROJECTION ANALYSIS: Where do BC-Nash deviations live?")
    print(f"{'=' * 60}")

    print(f"\nFraction of deviation norm in top-k SVD dimensions:")
    print(f"  {'k':>3}  {'P1 (mean)':>10}  {'P2 (mean)':>10}  {'Avg':>10}")
    for k in [1, 2, 3, 5, 10]:
        key_p = f"frac_p_top{k}"
        key_q = f"frac_q_top{k}"
        if key_p not in results[0]:
            continue
        fp = np.mean([r[key_p] for r in results])
        fq = np.mean([r[key_q] for r in results])
        print(f"  {k:>3}  {fp:>10.4f}  {fq:>10.4f}  {(fp + fq) / 2:>10.4f}")

    print(f"\nFraction of deviation ENERGY (squared) in top-k:")
    print(f"  {'k':>3}  {'P1 (mean)':>10}  {'P2 (mean)':>10}  {'Avg':>10}")
    for k in [1, 2, 3, 5, 10]:
        key_p = f"energy_p_top{k}"
        key_q = f"energy_q_top{k}"
        if key_p not in results[0]:
            continue
        ep = np.mean([r[key_p] for r in results])
        eq = np.mean([r[key_q] for r in results])
        print(f"  {k:>3}  {ep:>10.4f}  {eq:>10.4f}  {(ep + eq) / 2:>10.4f}")

    # Null space fractions
    if "frac_p_null3" in results[0]:
        null_p = np.mean([r["frac_p_null3"] for r in results])
        null_q = np.mean([r["frac_q_null3"] for r in results])
        print(f"\nFraction of deviation energy in NULL SPACE of top-3:")
        print(f"  P1: {null_p:.4f}")
        print(f"  P2: {null_q:.4f}")
        print(f"  Avg: {(null_p + null_q) / 2:.4f}")

    # Bootstrap CIs for key metrics
    boot_data = {}
    for k in [1, 2, 3, 5, 10]:
        key = f"energy_avg_top{k}"
        if key in results[0]:
            boot_data[key] = np.array([r[key] for r in results])
    boot = bootstrap_all(boot_data, n_resamples=10_000, seed=42)

    print(f"\nBootstrap CIs for average deviation energy in top-k:")
    for k in [1, 2, 3, 5, 10]:
        key = f"energy_avg_top{k}"
        if key not in boot:
            continue
        b = boot[key]
        print(
            f"  top-{k}: {b['mean']:.4f} [{b['mean_ci_lo']:.4f}, {b['mean_ci_hi']:.4f}]"
        )

    # Save
    output = {
        "n_matchups": n,
        "per_matchup": results,
        "aggregate": {
            "bootstrap": boot,
            "summary": {
                f"energy_avg_top{k}": float(
                    np.mean([r[f"energy_avg_top{k}"] for r in results])
                )
                for k in [1, 2, 3, 5, 10]
                if f"energy_avg_top{k}" in results[0]
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
