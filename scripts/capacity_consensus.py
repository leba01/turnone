"""Per-matchup rank consensus across dynamics model capacities.

Loads SVD results from three model capacities (d_action=32, 64, 128) and
computes per-matchup agreement statistics for effective rank at the 95%
energy threshold. Used to strengthen the capacity-robustness argument in §4.
"""

import json
import numpy as np
from scipy import stats
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results" / "phase2"

FILES = {
    32: RESULTS / "svd.json",
    64: RESULTS / "svd_d64.json",
    128: RESULTS / "svd_d128.json",
}


def load_ranks(path: Path) -> dict[int, int]:
    """Return {matchup_idx: eff_rank_95} from an SVD result file."""
    with open(path) as f:
        data = json.load(f)
    return {m["idx"]: m["eff_rank_95"] for m in data["per_matchup"]}


def load_top3(path: Path) -> dict[int, float]:
    """Return {matchup_idx: top_3_frac} from an SVD result file."""
    with open(path) as f:
        data = json.load(f)
    return {m["idx"]: m["top_3_frac"] for m in data["per_matchup"]}


def main():
    # Load per-matchup effective ranks
    ranks = {d: load_ranks(p) for d, p in FILES.items()}

    # Align on shared matchup indices
    shared = sorted(set.intersection(*(set(r.keys()) for r in ranks.values())))
    n = len(shared)
    print(f"Shared matchups: {n}")

    # Build aligned arrays
    r32 = np.array([ranks[32][i] for i in shared])
    r64 = np.array([ranks[64][i] for i in shared])
    r128 = np.array([ranks[128][i] for i in shared])

    # Spearman correlations
    pairs = [(32, 64, r32, r64), (32, 128, r32, r128), (64, 128, r64, r128)]
    print("\n--- Spearman correlation of eff_rank_95 ---")
    for d1, d2, a, b in pairs:
        rho, p = stats.spearmanr(a, b)
        print(f"  d{d1} vs d{d2}: ρ = {rho:.3f}  (p = {p:.2e})")

    # % of matchups where all 3 agree within ±1
    diffs_max = np.max(np.abs(np.column_stack([r32 - r64, r32 - r128, r64 - r128])), axis=1)
    agree_1 = np.mean(diffs_max <= 1) * 100
    print(f"\nAll-3 agree within ±1: {agree_1:.1f}%")

    # Median absolute deviation of eff_rank across capacities per matchup
    stacked = np.column_stack([r32, r64, r128])
    per_matchup_mad = np.median(np.abs(stacked - np.median(stacked, axis=1, keepdims=True)), axis=1)
    print(f"Median absolute deviation (across matchups): {np.median(per_matchup_mad):.2f}")
    print(f"Mean absolute deviation (across matchups): {np.mean(per_matchup_mad):.2f}")

    # Top-3 energy fraction correlation
    top3 = {d: load_top3(p) for d, p in FILES.items()}
    t32 = np.array([top3[32][i] for i in shared])
    t64 = np.array([top3[64][i] for i in shared])
    t128 = np.array([top3[128][i] for i in shared])

    print("\n--- Spearman correlation of top_3_frac ---")
    for d1, d2, a, b in [(32, 64, t32, t64), (32, 128, t32, t128), (64, 128, t64, t128)]:
        rho, p = stats.spearmanr(a, b)
        print(f"  d{d1} vs d{d2}: ρ = {rho:.3f}  (p = {p:.2e})")

    # Summary stats
    print("\n--- Summary ---")
    for d, arr in [(32, r32), (64, r64), (128, r128)]:
        print(f"  d{d}: median rank = {np.median(arr):.1f}, mean = {np.mean(arr):.1f}")


if __name__ == "__main__":
    main()
