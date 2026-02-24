"""Phase 2 Experiment 3: Indifference analysis.

Tests whether BC strategies approximately satisfy the indifference principle
(all support actions yield equal payoff) against the BC population vs Nash.
Measures ecological adaptation: is BC tuned to its niche?

Usage:
    python scripts/phase2_indifference.py \
        --cache results/phase2/cache.pkl \
        --out results/phase2/indifference.json
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from turnone.eval.bootstrap import bootstrap_all


def analyze_matchup(m: dict) -> dict:
    """Indifference analysis for one matchup.

    Computes 2x2 table: {BC, Nash} strategy × {BC, Nash} opponent.
    For each cell: weighted payoff variance, max gap in support,
    fraction of mass on epsilon-best-response.
    """
    R = np.asarray(m["R"], dtype=np.float64)
    bc_p1 = np.asarray(m["bc_p1"], dtype=np.float64)
    bc_p2 = np.asarray(m["bc_p2"], dtype=np.float64)
    nash_p1 = np.asarray(m["nash_p1"], dtype=np.float64)
    nash_p2 = np.asarray(m["nash_p2"], dtype=np.float64)
    game_value = m["game_value"]

    strategies = {"bc": bc_p1, "nash": nash_p1}
    opponents = {"bc": bc_p2, "nash": nash_p2}

    result = {"idx": m["idx"], "game_value": game_value}
    support_threshold = 1e-3

    for strat_name, strat in strategies.items():
        for opp_name, opp in opponents.items():
            prefix = f"{strat_name}_vs_{opp_name}"

            # Action payoffs for P1 against this opponent
            payoffs = R @ opp  # (n1,)

            # Value under this strategy-opponent pair
            value = float(strat @ payoffs)

            # Weighted payoff variance: Var = sum_i p_i (v_i - V_bar)^2
            weighted_var = float(np.sum(strat * (payoffs - value) ** 2))
            weighted_std = float(np.sqrt(weighted_var))

            # Support analysis
            support_mask = strat > support_threshold
            n_support = int(support_mask.sum())

            if n_support > 0:
                support_payoffs = payoffs[support_mask]
                max_gap = float(support_payoffs.max() - support_payoffs.min())
            else:
                max_gap = 0.0

            # Fraction of mass on epsilon-best-response
            best_payoff = float(payoffs.max())
            eps_mass = {}
            for eps in [0.1, 0.5, 1.0]:
                mask = payoffs >= best_payoff - eps
                eps_mass[f"eps_{str(eps).replace('.', 'p')}_mass"] = float(strat[mask].sum())

            result[f"{prefix}_value"] = value
            result[f"{prefix}_weighted_var"] = weighted_var
            result[f"{prefix}_weighted_std"] = weighted_std
            result[f"{prefix}_max_gap"] = max_gap
            result[f"{prefix}_n_support"] = n_support
            result.update({f"{prefix}_{k}": v for k, v in eps_mass.items()})

    # Indifference ratio: Var(BC vs BC) / Var(BC vs Nash)
    var_bc_bc = result["bc_vs_bc_weighted_var"]
    var_bc_nash = result["bc_vs_nash_weighted_var"]
    if var_bc_nash > 1e-10:
        result["indifference_ratio"] = var_bc_bc / var_bc_nash
    else:
        result["indifference_ratio"] = float("nan")

    return result


def main():
    parser = argparse.ArgumentParser(description="Phase 2 Exp 3: Indifference analysis")
    parser.add_argument("--cache", required=True, help="Path to cache.pkl")
    parser.add_argument("--out", required=True, help="Output JSON path")
    args = parser.parse_args()

    with open(args.cache, "rb") as f:
        matchups = pickle.load(f)
    print(f"Loaded {len(matchups)} matchups from cache")

    results = [analyze_matchup(m) for m in matchups]

    # Aggregate: 2x2 table
    print(f"\n{'='*70}")
    print(f"INDIFFERENCE ANALYSIS ({len(results)} matchups)")
    print(f"{'='*70}")

    print(f"\nWeighted payoff std (mean ± std across matchups):")
    print(f"{'':>20} {'vs BC opp':>15} {'vs Nash opp':>15}")
    print(f"{'-'*50}")

    for strat_name in ["bc", "nash"]:
        stds_bc = np.array([r[f"{strat_name}_vs_bc_weighted_std"] for r in results])
        stds_nash = np.array([r[f"{strat_name}_vs_nash_weighted_std"] for r in results])
        label = "BC strategy" if strat_name == "bc" else "Nash strategy"
        print(f"{label:>20} {stds_bc.mean():>7.4f}±{stds_bc.std():>5.4f}  "
              f"{stds_nash.mean():>7.4f}±{stds_nash.std():>5.4f}")

    # Indifference ratio
    ratios = np.array([r["indifference_ratio"] for r in results])
    finite_ratios = ratios[np.isfinite(ratios)]
    print(f"\nIndifference ratio (Var_BC_vs_BC / Var_BC_vs_Nash):")
    print(f"  mean: {finite_ratios.mean():.4f}, median: {np.median(finite_ratios):.4f}")
    print(f"  fraction < 1 (BC more indifferent vs BC): {(finite_ratios < 1).mean():.1%}")

    # Epsilon-BR mass
    print(f"\nFraction of BC mass on ε-best-response:")
    for eps_key in ["eps_0p1_mass", "eps_0p5_mass", "eps_1p0_mass"]:
        eps_label = eps_key.replace("eps_", "ε=").replace("p", ".").replace("_mass", "")
        mass_bc = np.array([r[f"bc_vs_bc_{eps_key}"] for r in results])
        mass_nash = np.array([r[f"bc_vs_nash_{eps_key}"] for r in results])
        print(f"  {eps_label}: vs BC {mass_bc.mean():.3f}, vs Nash {mass_nash.mean():.3f}")

    # Nash vs Nash sanity check
    nash_nash_std = np.array([r["nash_vs_nash_weighted_std"] for r in results])
    print(f"\nSanity check — Nash vs Nash weighted std: "
          f"mean {nash_nash_std.mean():.6f} (should be ~0)")

    # Max gap in support
    print(f"\nMax payoff gap within support:")
    for strat_name in ["bc", "nash"]:
        for opp_name in ["bc", "nash"]:
            gaps = np.array([r[f"{strat_name}_vs_{opp_name}_max_gap"] for r in results])
            label = f"{strat_name} vs {opp_name}"
            print(f"  {label:>15}: mean {gaps.mean():.4f}, median {np.median(gaps):.4f}")

    # Bootstrap CIs
    boot_data = {
        "bc_vs_bc_std": np.array([r["bc_vs_bc_weighted_std"] for r in results]),
        "bc_vs_nash_std": np.array([r["bc_vs_nash_weighted_std"] for r in results]),
        "nash_vs_bc_std": np.array([r["nash_vs_bc_weighted_std"] for r in results]),
        "nash_vs_nash_std": nash_nash_std,
        "indifference_ratio": finite_ratios,
    }
    boot_results = bootstrap_all(boot_data, n_resamples=10_000, seed=42)

    # Save
    output = {
        "n_matchups": len(results),
        "per_matchup": results,
        "aggregate": {
            "table_weighted_std": {
                f"{s}_vs_{o}": {
                    "mean": float(np.mean([r[f"{s}_vs_{o}_weighted_std"] for r in results])),
                    "median": float(np.median([r[f"{s}_vs_{o}_weighted_std"] for r in results])),
                }
                for s in ["bc", "nash"]
                for o in ["bc", "nash"]
            },
            "indifference_ratio": {
                "mean": float(finite_ratios.mean()),
                "median": float(np.median(finite_ratios)),
                "frac_lt_1": float((finite_ratios < 1).mean()),
            },
            "bootstrap": boot_results,
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
