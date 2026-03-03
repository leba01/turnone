"""Tera decomposition: how much exploitability comes from tera underuse?

BR uses tera 94% vs BC's 25%. This script isolates the tera contribution to
exploitability by computing BR restricted to the same tera class as BC.

Works from cache.pkl only (no dataset needed). Actions are enumerated as
(slot_a, slot_b, tera_flag) with tera as the innermost loop, so
action index i has tera_flag = i % 3 (0=none, 1=tera_A, 2=tera_B).

For each matchup:
  - Compute full BR (unrestricted): argmax of R @ bc_p2
  - Compute tera-restricted BR: only allow actions in same tera class as BC's mode
  - Difference = "tera contribution to exploitability"

Usage:
    python scripts/tera_decomposition.py \
        --cache results/phase2/cache.pkl \
        --out results/tera_decomposition/tera_decomposition.json
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from turnone.eval.bootstrap import bootstrap_all


TERA_NAMES = ["none", "tera_A", "tera_B"]


def analyze_matchup(m: dict) -> dict:
    """Decompose exploitability into tera-dependent and tera-independent parts."""
    R = np.asarray(m["R"], dtype=np.float64)
    bc_p1 = np.asarray(m["bc_p1"], dtype=np.float64)
    bc_p2 = np.asarray(m["bc_p2"], dtype=np.float64)
    n1 = R.shape[0]

    # Tera flags from action enumeration structure
    tera_flags = np.array([i % 3 for i in range(n1)])

    # BC-vs-BC payoff
    bc_vs_bc = float(bc_p1 @ R @ bc_p2)

    # Full BR payoffs (per row)
    br_payoffs = R @ bc_p2  # (n1,)
    br_idx_full = int(np.argmax(br_payoffs))
    br_payoff_full = float(br_payoffs[br_idx_full])
    exploit_full = br_payoff_full - bc_vs_bc

    # BC's tera mode: weighted tera distribution
    bc_tera_mass = np.zeros(3)
    for tf in range(3):
        bc_tera_mass[tf] = bc_p1[tera_flags == tf].sum()
    bc_tera_mode = int(np.argmax(bc_tera_mass))

    # BR's tera choice
    br_tera_full = tera_flags[br_idx_full]

    # Tera-restricted BR: only allow actions with same tera as BC's mode
    tera_mask = tera_flags == bc_tera_mode

    restricted_payoffs = br_payoffs.copy()
    restricted_payoffs[~tera_mask] = -np.inf
    br_idx_restricted = int(np.argmax(restricted_payoffs))
    br_payoff_restricted = float(br_payoffs[br_idx_restricted])
    exploit_restricted = br_payoff_restricted - bc_vs_bc

    # Tera contribution
    tera_contribution = exploit_full - exploit_restricted
    tera_frac = tera_contribution / exploit_full if exploit_full > 1e-10 else 0.0

    # BR tera usage across all matchups (for aggregation)
    # Also: BC weighted tera distribution (for comparison)
    bc_uses_tera = float(1.0 - bc_tera_mass[0])  # fraction using any tera
    br_uses_tera = br_tera_full != 0

    return {
        "idx": m["idx"],
        "exploit_full": float(exploit_full),
        "exploit_tera_restricted": float(exploit_restricted),
        "tera_contribution": float(tera_contribution),
        "tera_frac_of_exploit": float(tera_frac),
        "bc_tera_mode": TERA_NAMES[bc_tera_mode],
        "bc_tera_mass": {TERA_NAMES[i]: float(bc_tera_mass[i]) for i in range(3)},
        "bc_uses_tera_frac": float(bc_uses_tera),
        "br_tera": TERA_NAMES[br_tera_full],
        "br_uses_tera": bool(br_uses_tera),
        "br_changed_tera": int(br_tera_full) != bc_tera_mode,
    }


def main():
    parser = argparse.ArgumentParser(description="Tera decomposition")
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
    print("TERA DECOMPOSITION: How much exploit comes from tera?")
    print(f"{'=' * 60}")

    exploit_full = np.array([r["exploit_full"] for r in results])
    exploit_restricted = np.array([r["exploit_tera_restricted"] for r in results])
    tera_contrib = np.array([r["tera_contribution"] for r in results])
    tera_frac = np.array([r["tera_frac_of_exploit"] for r in results])
    changed_tera = np.array([r["br_changed_tera"] for r in results])
    bc_uses_tera = np.array([r["bc_uses_tera_frac"] for r in results])
    br_uses_tera = np.array([r["br_uses_tera"] for r in results])

    print(
        f"\nFull exploit:            {np.mean(exploit_full):.4f} +/- {np.std(exploit_full):.4f}"
    )
    print(
        f"Tera-restricted exploit: {np.mean(exploit_restricted):.4f} +/- {np.std(exploit_restricted):.4f}"
    )
    print(
        f"Tera contribution:       {np.mean(tera_contrib):.4f} +/- {np.std(tera_contrib):.4f}"
    )
    print(
        f"Tera fraction of total:  {np.mean(tera_frac):.4f} ({np.mean(tera_frac) * 100:.1f}%)"
    )
    print(f"BR changed tera:         {np.mean(changed_tera) * 100:.1f}%")
    print(f"BC uses tera (weighted): {np.mean(bc_uses_tera) * 100:.1f}%")
    print(f"BR uses tera:            {np.mean(br_uses_tera) * 100:.1f}%")

    # BC tera mode distribution
    bc_modes = [r["bc_tera_mode"] for r in results]
    for t in TERA_NAMES:
        frac = sum(1 for m in bc_modes if m == t) / n
        print(f"  BC tera mode = {t}: {frac * 100:.1f}%")

    # Bootstrap CIs
    boot_data = {
        "exploit_full": exploit_full,
        "exploit_tera_restricted": exploit_restricted,
        "tera_contribution": tera_contrib,
        "tera_frac_of_exploit": tera_frac,
    }
    boot = bootstrap_all(boot_data, n_resamples=10_000, seed=42)

    print(f"\nBootstrap CIs:")
    for key in [
        "exploit_full",
        "exploit_tera_restricted",
        "tera_contribution",
        "tera_frac_of_exploit",
    ]:
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
                "mean_exploit_full": float(np.mean(exploit_full)),
                "mean_exploit_restricted": float(np.mean(exploit_restricted)),
                "mean_tera_contribution": float(np.mean(tera_contrib)),
                "mean_tera_frac": float(np.mean(tera_frac)),
                "br_changed_tera_frac": float(np.mean(changed_tera)),
                "bc_uses_tera_mean": float(np.mean(bc_uses_tera)),
                "br_uses_tera_frac": float(np.mean(br_uses_tera)),
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
