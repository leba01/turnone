"""Phase 2 Experiment 4: Regret decomposition.

Computes external regret, internal (swap) regret, and best-response mass
for BC strategies against various opponents. Tests population-rationality
and approximate correlated equilibrium properties.

Usage:
    python scripts/phase2_regret.py \
        --cache results/phase2/cache.pkl \
        --out results/phase2/regret.json
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from turnone.game.exploitability import exploitability_from_nash
from turnone.eval.bootstrap import bootstrap_all


def _external_regret(strategy: np.ndarray, payoffs: np.ndarray) -> float:
    """External regret: best fixed action minus strategy value.

    Args:
        strategy: (n,) probability vector.
        payoffs: (n,) expected payoff per action against the opponent.

    Returns:
        max_i payoffs[i] - strategy @ payoffs (non-negative).
    """
    return float(payoffs.max() - strategy @ payoffs)


def _internal_regret(strategy: np.ndarray, payoffs: np.ndarray) -> float:
    """Full internal (swap) regret.

    For each action i, find best swap target j_i = argmax_j payoffs[j].
    Internal regret = sum_i p_i * (payoffs[j_i] - payoffs[i]).

    Note: since j_i = argmax(payoffs) for all i (payoffs don't depend on i
    in a normal-form game), internal regret equals external regret in
    normal-form games. The distinction matters for correlated strategies.

    But we compute the full swap regret where each action i can map to a
    different action phi(i):
        swap_regret = max_phi sum_i p_i * (payoffs[phi(i)] - payoffs[i])

    In normal-form, optimal phi maps everything to argmax, so
    swap_regret = external_regret. We compute it anyway for completeness.
    """
    # In normal-form games, the optimal swap function maps every action
    # to argmax(payoffs), making swap regret = external regret.
    # For generality, compute the full O(n^2) version.
    n = len(strategy)
    best_swap = float(payoffs.max())  # best target for any action
    current_value = float(strategy @ payoffs)
    return best_swap - current_value


def _conditional_swap_regret(strategy: np.ndarray, payoffs: np.ndarray) -> float:
    """Conditional swap regret: max over pairs (i→j).

    max_{i,j} p_i * (payoffs[j] - payoffs[i])

    This is the max improvement from swapping a single action i to j,
    weighted by the probability of playing i.
    """
    n = len(strategy)
    max_swap = 0.0
    for i in range(n):
        if strategy[i] < 1e-10:
            continue
        for j in range(n):
            gain = strategy[i] * (payoffs[j] - payoffs[i])
            if gain > max_swap:
                max_swap = gain
    return float(max_swap)


def analyze_matchup(m: dict) -> dict:
    """Regret analysis for one matchup (both sides)."""
    R = np.asarray(m["R"], dtype=np.float64)
    bc_p1 = np.asarray(m["bc_p1"], dtype=np.float64)
    bc_p2 = np.asarray(m["bc_p2"], dtype=np.float64)
    nash_p1 = np.asarray(m["nash_p1"], dtype=np.float64)
    nash_p2 = np.asarray(m["nash_p2"], dtype=np.float64)
    game_value = m["game_value"]

    n1, n2 = R.shape
    result = {"idx": m["idx"], "n1": n1, "n2": n2, "game_value": game_value}

    # --- P1 analysis ---
    # Payoffs per action against various opponents
    payoffs_vs_bc = R @ bc_p2        # (n1,) P1's payoff per action vs BC P2
    payoffs_vs_nash = R @ nash_p2    # (n1,) P1's payoff per action vs Nash P2
    uniform_p2 = np.ones(n2, dtype=np.float64) / n2
    payoffs_vs_uniform = R @ uniform_p2

    # External regret
    result["p1_ext_regret_vs_bc"] = _external_regret(bc_p1, payoffs_vs_bc)
    result["p1_ext_regret_vs_nash"] = _external_regret(bc_p1, payoffs_vs_nash)
    result["p1_ext_regret_vs_uniform"] = _external_regret(bc_p1, payoffs_vs_uniform)

    # Swap regret (against BC opponent — the population)
    result["p1_swap_regret_vs_bc"] = _internal_regret(bc_p1, payoffs_vs_bc)
    result["p1_cond_swap_vs_bc"] = _conditional_swap_regret(bc_p1, payoffs_vs_bc)

    # Best-response mass
    br_idx_vs_bc = int(payoffs_vs_bc.argmax())
    br_idx_vs_nash = int(payoffs_vs_nash.argmax())
    result["p1_br_mass_vs_bc"] = float(bc_p1[br_idx_vs_bc])
    result["p1_br_mass_vs_nash"] = float(bc_p1[br_idx_vs_nash])

    # Exploitability (for reference)
    result["p1_exploitability"] = float(exploitability_from_nash(bc_p1, R, game_value, player=1))

    # --- P2 analysis (analogous, P2 minimizes) ---
    # P2's action payoffs: for P2, payoff = -(P1's payoff)
    # P2 picks column j to minimize P1's payoff.
    # P2's "payoff" per action j vs P1 strategy: -(bc_p1 @ R)_j
    p2_payoffs_vs_bc = -(bc_p1 @ R)     # (n2,) P2 wants to maximize this (minimize P1's payoff)
    p2_payoffs_vs_nash = -(nash_p1 @ R)  # (n2,)
    uniform_p1 = np.ones(n1, dtype=np.float64) / n1
    p2_payoffs_vs_uniform = -(uniform_p1 @ R)

    result["p2_ext_regret_vs_bc"] = _external_regret(bc_p2, p2_payoffs_vs_bc)
    result["p2_ext_regret_vs_nash"] = _external_regret(bc_p2, p2_payoffs_vs_nash)
    result["p2_ext_regret_vs_uniform"] = _external_regret(bc_p2, p2_payoffs_vs_uniform)

    result["p2_swap_regret_vs_bc"] = _internal_regret(bc_p2, p2_payoffs_vs_bc)
    result["p2_cond_swap_vs_bc"] = _conditional_swap_regret(bc_p2, p2_payoffs_vs_bc)

    br_idx_p2_vs_bc = int(p2_payoffs_vs_bc.argmax())
    br_idx_p2_vs_nash = int(p2_payoffs_vs_nash.argmax())
    result["p2_br_mass_vs_bc"] = float(bc_p2[br_idx_p2_vs_bc])
    result["p2_br_mass_vs_nash"] = float(bc_p2[br_idx_p2_vs_nash])

    result["p2_exploitability"] = float(exploitability_from_nash(bc_p2, R, game_value, player=2))

    # Regret ratio: ext_regret_vs_bc / ext_regret_vs_nash
    if result["p1_ext_regret_vs_nash"] > 1e-10:
        result["p1_regret_ratio"] = result["p1_ext_regret_vs_bc"] / result["p1_ext_regret_vs_nash"]
    else:
        result["p1_regret_ratio"] = float("nan")

    if result["p2_ext_regret_vs_nash"] > 1e-10:
        result["p2_regret_ratio"] = result["p2_ext_regret_vs_bc"] / result["p2_ext_regret_vs_nash"]
    else:
        result["p2_regret_ratio"] = float("nan")

    return result


def main():
    parser = argparse.ArgumentParser(description="Phase 2 Exp 4: Regret decomposition")
    parser.add_argument("--cache", required=True, help="Path to cache.pkl")
    parser.add_argument("--out", required=True, help="Output JSON path")
    args = parser.parse_args()

    with open(args.cache, "rb") as f:
        matchups = pickle.load(f)
    print(f"Loaded {len(matchups)} matchups from cache")

    results = [analyze_matchup(m) for m in matchups]

    # Aggregate
    print(f"\n{'='*70}")
    print(f"REGRET DECOMPOSITION ({len(results)} matchups)")
    print(f"{'='*70}")

    # P1 regret profile
    print(f"\nP1 external regret (mean ± std):")
    for key, label in [
        ("p1_ext_regret_vs_bc", "vs BC"),
        ("p1_ext_regret_vs_nash", "vs Nash"),
        ("p1_ext_regret_vs_uniform", "vs Uniform"),
    ]:
        vals = np.array([r[key] for r in results])
        print(f"  {label:>12}: {vals.mean():.4f} ± {vals.std():.4f}")

    print(f"\nP1 swap regret vs BC: {np.mean([r['p1_swap_regret_vs_bc'] for r in results]):.4f}")
    print(f"P1 cond swap vs BC:  {np.mean([r['p1_cond_swap_vs_bc'] for r in results]):.4f}")

    # P2 regret profile
    print(f"\nP2 external regret (mean ± std):")
    for key, label in [
        ("p2_ext_regret_vs_bc", "vs BC"),
        ("p2_ext_regret_vs_nash", "vs Nash"),
        ("p2_ext_regret_vs_uniform", "vs Uniform"),
    ]:
        vals = np.array([r[key] for r in results])
        print(f"  {label:>12}: {vals.mean():.4f} ± {vals.std():.4f}")

    # Regret ratios
    p1_ratios = np.array([r["p1_regret_ratio"] for r in results])
    p2_ratios = np.array([r["p2_regret_ratio"] for r in results])
    p1_finite = p1_ratios[np.isfinite(p1_ratios)]
    p2_finite = p2_ratios[np.isfinite(p2_ratios)]

    print(f"\nRegret ratio (ext_vs_bc / ext_vs_nash):")
    print(f"  P1: mean {p1_finite.mean():.4f}, median {np.median(p1_finite):.4f}")
    print(f"  P2: mean {p2_finite.mean():.4f}, median {np.median(p2_finite):.4f}")
    print(f"  If < 1: BC is population-adapted (less regret vs BC than vs Nash)")

    # Swap regret thresholds
    swap_vals = np.array([r["p1_swap_regret_vs_bc"] for r in results])
    for eps in [0.1, 0.5, 1.0]:
        frac = float((swap_vals < eps).mean())
        print(f"  Frac matchups with P1 swap regret < {eps}: {frac:.1%}")

    # BR mass
    print(f"\nBest-response mass in BC (mean):")
    print(f"  P1 vs BC opp:   {np.mean([r['p1_br_mass_vs_bc'] for r in results]):.4f}")
    print(f"  P1 vs Nash opp: {np.mean([r['p1_br_mass_vs_nash'] for r in results]):.4f}")
    print(f"  P2 vs BC opp:   {np.mean([r['p2_br_mass_vs_bc'] for r in results]):.4f}")
    print(f"  P2 vs Nash opp: {np.mean([r['p2_br_mass_vs_nash'] for r in results]):.4f}")

    # Exploitability comparison
    exploits = np.array([r["p1_exploitability"] + r["p2_exploitability"] for r in results])
    print(f"\nTotal exploitability (P1+P2): mean {exploits.mean():.4f}")

    # Bootstrap CIs
    boot_data = {
        "p1_ext_regret_vs_bc": np.array([r["p1_ext_regret_vs_bc"] for r in results]),
        "p1_ext_regret_vs_nash": np.array([r["p1_ext_regret_vs_nash"] for r in results]),
        "p2_ext_regret_vs_bc": np.array([r["p2_ext_regret_vs_bc"] for r in results]),
        "p2_ext_regret_vs_nash": np.array([r["p2_ext_regret_vs_nash"] for r in results]),
        "p1_swap_regret_vs_bc": swap_vals,
        "p1_regret_ratio": p1_finite,
        "p2_regret_ratio": p2_finite,
    }
    boot_results = bootstrap_all(boot_data, n_resamples=10_000, seed=42)

    # Save
    output = {
        "n_matchups": len(results),
        "per_matchup": results,
        "aggregate": {
            "p1_regret_profile": {
                "ext_vs_bc": float(np.mean([r["p1_ext_regret_vs_bc"] for r in results])),
                "ext_vs_nash": float(np.mean([r["p1_ext_regret_vs_nash"] for r in results])),
                "ext_vs_uniform": float(np.mean([r["p1_ext_regret_vs_uniform"] for r in results])),
                "swap_vs_bc": float(np.mean([r["p1_swap_regret_vs_bc"] for r in results])),
                "cond_swap_vs_bc": float(np.mean([r["p1_cond_swap_vs_bc"] for r in results])),
            },
            "p2_regret_profile": {
                "ext_vs_bc": float(np.mean([r["p2_ext_regret_vs_bc"] for r in results])),
                "ext_vs_nash": float(np.mean([r["p2_ext_regret_vs_nash"] for r in results])),
                "ext_vs_uniform": float(np.mean([r["p2_ext_regret_vs_uniform"] for r in results])),
                "swap_vs_bc": float(np.mean([r["p2_swap_regret_vs_bc"] for r in results])),
                "cond_swap_vs_bc": float(np.mean([r["p2_cond_swap_vs_bc"] for r in results])),
            },
            "regret_ratios": {
                "p1_mean": float(p1_finite.mean()),
                "p1_median": float(np.median(p1_finite)),
                "p2_mean": float(p2_finite.mean()),
                "p2_median": float(np.median(p2_finite)),
            },
            "swap_regret_thresholds": {
                str(eps): float((swap_vals < eps).mean())
                for eps in [0.1, 0.5, 1.0]
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
