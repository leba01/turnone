"""Interaction term analysis: connecting Eq. 1 to the low-rank structure.

Computes the bilinear decomposition (p-p*)^T R (q-q*) directly from cache
and connects it to the projection analysis: the interaction is small precisely
because deviations are orthogonal to R's column space.

Usage:
    python scripts/interaction_term.py \
        --cache results/phase2/cache.pkl \
        --out results/interaction/interaction.json
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from turnone.eval.bootstrap import bootstrap_all


def analyze_matchup(m: dict) -> dict:
    """Compute Eq. 1 decomposition for a single matchup."""
    R = np.asarray(m["R"], dtype=np.float64)
    bc_p1 = np.asarray(m["bc_p1"], dtype=np.float64)
    bc_p2 = np.asarray(m["bc_p2"], dtype=np.float64)
    nash_p1 = np.asarray(m["nash_p1"], dtype=np.float64)
    nash_p2 = np.asarray(m["nash_p2"], dtype=np.float64)
    game_value = m["game_value"]

    # BC-vs-BC payoff
    bc_vs_bc = float(bc_p1 @ R @ bc_p2)

    # Full gap
    total_gap = bc_vs_bc - game_value

    # Delta_1: p*^T R (q_bc - q*) <= 0
    delta_q = bc_p2 - nash_p2
    delta_1 = float(nash_p1 @ R @ delta_q)

    # Delta_2: (p_bc - p*)^T R q* >= 0
    delta_p = bc_p1 - nash_p1
    delta_2 = float(delta_p @ R @ nash_p2)

    # Interaction: (p_bc - p*)^T R (q_bc - q*)
    interaction = float(delta_p @ R @ delta_q)

    # Verification: total_gap should equal delta_1 + delta_2 + interaction
    reconstruction = delta_1 + delta_2 + interaction
    reconstruction_error = abs(total_gap - reconstruction)

    # Sign check
    delta_1_sign = (
        "negative" if delta_1 < -1e-10 else ("positive" if delta_1 > 1e-10 else "zero")
    )
    delta_2_sign = (
        "positive" if delta_2 > 1e-10 else ("negative" if delta_2 < -1e-10 else "zero")
    )
    opposite_signs = (delta_1 < -1e-10 and delta_2 > 1e-10) or (
        delta_1 > 1e-10 and delta_2 < -1e-10
    )

    # Cancellation: how well do delta_1 and delta_2 cancel?
    sum_d1_d2 = delta_1 + delta_2
    cancellation_ratio = abs(sum_d1_d2) / max(abs(delta_1), abs(delta_2), 1e-15)

    return {
        "idx": m["idx"],
        "bc_vs_bc": bc_vs_bc,
        "game_value": game_value,
        "total_gap": total_gap,
        "delta_1": delta_1,
        "delta_2": delta_2,
        "interaction": interaction,
        "reconstruction_error": reconstruction_error,
        "delta_1_sign": delta_1_sign,
        "delta_2_sign": delta_2_sign,
        "opposite_signs": opposite_signs,
        "sum_d1_d2": sum_d1_d2,
        "cancellation_ratio": cancellation_ratio,
    }


def main():
    parser = argparse.ArgumentParser(description="Interaction term analysis")
    parser.add_argument("--cache", required=True, help="Path to cache.pkl")
    parser.add_argument("--out", required=True, help="Output JSON path")
    args = parser.parse_args()

    with open(args.cache, "rb") as f:
        matchups = pickle.load(f)
    print(f"Loaded {len(matchups)} matchups from cache")

    results = [analyze_matchup(m) for m in matchups]
    n = len(results)

    # Summary
    print(f"\n{'=' * 60}")
    print("INTERACTION TERM: Eq. 1 decomposition")
    print(f"{'=' * 60}")

    delta_1 = np.array([r["delta_1"] for r in results])
    delta_2 = np.array([r["delta_2"] for r in results])
    interaction = np.array([r["interaction"] for r in results])
    total_gap = np.array([r["total_gap"] for r in results])
    sum_d1_d2 = np.array([r["sum_d1_d2"] for r in results])
    opposite_signs = np.array([r["opposite_signs"] for r in results])
    cancel_ratio = np.array([r["cancellation_ratio"] for r in results])

    print(
        f"\nΔ1 (p* exploit of q_bc):  mean={np.mean(delta_1):.4f}, std={np.std(delta_1):.4f}"
    )
    print(
        f"Δ2 (p_bc exploit of q*):  mean={np.mean(delta_2):.4f}, std={np.std(delta_2):.4f}"
    )
    print(
        f"Interaction (δp^T R δq):  mean={np.mean(interaction):.4f}, std={np.std(interaction):.4f}"
    )
    print(
        f"Total gap (bc-bc - V*):   mean={np.mean(total_gap):.4f}, std={np.std(total_gap):.4f}"
    )
    print(
        f"\nΔ1 + Δ2:                 mean={np.mean(sum_d1_d2):.4f}, std={np.std(sum_d1_d2):.4f}"
    )
    print(
        f"Opposite signs:           {np.sum(opposite_signs)}/{n} ({np.mean(opposite_signs) * 100:.1f}%)"
    )
    print(
        f"Cancellation ratio:       mean={np.mean(cancel_ratio):.4f}, median={np.median(cancel_ratio):.4f}"
    )

    # Reconstruction check
    recon_errors = np.array([r["reconstruction_error"] for r in results])
    print(f"\nReconstruction error:     max={np.max(recon_errors):.2e} (should be ~0)")

    # Sign analysis
    n_d1_neg = np.sum(delta_1 < -1e-10)
    n_d2_pos = np.sum(delta_2 > 1e-10)
    print(
        f"\nΔ1 < 0: {n_d1_neg}/{n} ({n_d1_neg / n * 100:.1f}%) [expected from minimax theorem]"
    )
    print(
        f"Δ2 > 0: {n_d2_pos}/{n} ({n_d2_pos / n * 100:.1f}%) [expected from minimax theorem]"
    )

    # Bootstrap CIs
    boot_data = {
        "delta_1": delta_1,
        "delta_2": delta_2,
        "interaction": interaction,
        "total_gap": total_gap,
        "sum_d1_d2": sum_d1_d2,
    }
    boot = bootstrap_all(boot_data, n_resamples=10_000, seed=42)

    print(f"\nBootstrap CIs:")
    for key in ["delta_1", "delta_2", "interaction", "total_gap", "sum_d1_d2"]:
        b = boot[key]
        print(
            f"  {key}: {b['mean']:.4f} [{b['mean_ci_lo']:.4f}, {b['mean_ci_hi']:.4f}]"
        )

    # Save
    output = {
        "n_matchups": n,
        "per_matchup": results,
        "aggregate": {
            "bootstrap": boot,
            "summary": {
                "mean_delta_1": float(np.mean(delta_1)),
                "mean_delta_2": float(np.mean(delta_2)),
                "mean_interaction": float(np.mean(interaction)),
                "mean_total_gap": float(np.mean(total_gap)),
                "n_opposite_signs": int(np.sum(opposite_signs)),
                "frac_opposite_signs": float(np.mean(opposite_signs)),
                "mean_cancellation_ratio": float(np.mean(cancel_ratio)),
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
