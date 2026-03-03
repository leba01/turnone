"""Rank reduction test: does projecting R to rank-k preserve Nash and exploitability?

If the game is truly ~3-dimensional, then R_k (the rank-k approximation of R)
should yield similar Nash equilibria and exploitability for k >= 3.

For each matchup:
  - Compute rank-k approximation: R_k = U[:,:k] @ diag(S[:k]) @ Vt[:k,:]
  - Solve Nash on R_k via LP
  - Compute exploitability of BC on R_k
  - Compare Nash strategies (TV) and exploitability between R and R_k

Usage:
    python scripts/rank_reduction_test.py \
        --cache results/phase2/cache.pkl \
        --out results/rank_reduction/rank_reduction.json
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from turnone.game.nash import solve_nash_lp
from turnone.eval.bootstrap import bootstrap_all


def analyze_matchup(m: dict, ks: list[int]) -> dict:
    """Compare full-rank and reduced-rank games for a single matchup."""
    R = np.asarray(m["R"], dtype=np.float64)
    bc_p1 = np.asarray(m["bc_p1"], dtype=np.float64)
    bc_p2 = np.asarray(m["bc_p2"], dtype=np.float64)
    nash_p1 = np.asarray(m["nash_p1"], dtype=np.float64)
    nash_p2 = np.asarray(m["nash_p2"], dtype=np.float64)
    game_value = m["game_value"]

    n1, n2 = R.shape
    U, S, Vt = np.linalg.svd(R, full_matrices=False)

    # Full-rank exploitability (baseline)
    br_payoffs_full = R @ bc_p2
    exploit_full = float(np.max(br_payoffs_full)) - float(bc_p1 @ R @ bc_p2)

    result = {
        "idx": m["idx"],
        "n1": n1,
        "n2": n2,
        "exploit_full": exploit_full,
        "game_value_full": game_value,
        "rank_results": {},
    }

    max_k = min(R.shape)
    for k in ks:
        if k > max_k:
            continue

        # Rank-k approximation
        R_k = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]

        # Solve Nash on R_k
        try:
            nash_p1_k, nash_p2_k, gv_k = solve_nash_lp(R_k)
        except ValueError:
            result["rank_results"][str(k)] = {"status": "lp_failed"}
            continue

        # Exploitability of BC on R_k
        br_payoffs_k = R_k @ bc_p2
        exploit_k = float(np.max(br_payoffs_k)) - float(bc_p1 @ R_k @ bc_p2)

        # TV distance between full and reduced Nash
        tv_nash_p1 = 0.5 * float(np.abs(nash_p1 - nash_p1_k).sum())
        tv_nash_p2 = 0.5 * float(np.abs(nash_p2 - nash_p2_k).sum())

        # Exploit change
        exploit_ratio = (
            exploit_k / exploit_full if exploit_full > 1e-10 else float("inf")
        )

        result["rank_results"][str(k)] = {
            "exploit_k": float(exploit_k),
            "exploit_ratio": float(exploit_ratio),
            "game_value_k": float(gv_k),
            "game_value_diff": float(gv_k - game_value),
            "tv_nash_p1": tv_nash_p1,
            "tv_nash_p2": tv_nash_p2,
            "tv_nash_avg": (tv_nash_p1 + tv_nash_p2) / 2,
        }

    return result


def main():
    parser = argparse.ArgumentParser(description="Rank reduction test")
    parser.add_argument("--cache", required=True, help="Path to cache.pkl")
    parser.add_argument("--out", required=True, help="Output JSON path")
    args = parser.parse_args()

    with open(args.cache, "rb") as f:
        matchups = pickle.load(f)
    print(f"Loaded {len(matchups)} matchups from cache")

    ks = [1, 2, 3, 5, 10, 20]
    results = [analyze_matchup(m, ks) for m in matchups]
    n = len(results)

    # Summary
    print(f"\n{'=' * 60}")
    print("RANK REDUCTION TEST: Does low rank preserve the game?")
    print(f"{'=' * 60}")

    # For each k, aggregate exploit ratios and Nash TV
    print(
        f"\n{'k':>3}  {'Exploit ratio':>14}  {'|ΔV*|':>8}  {'TV(Nash)':>10}  {'N valid':>8}"
    )
    for k in ks:
        ratios = []
        gv_diffs = []
        tv_nashs = []
        for r in results:
            rk = r["rank_results"].get(str(k))
            if rk and rk.get("status") != "lp_failed":
                ratios.append(rk["exploit_ratio"])
                gv_diffs.append(abs(rk["game_value_diff"]))
                tv_nashs.append(rk["tv_nash_avg"])

        if not ratios:
            continue

        ratios = np.array(ratios)
        gv_diffs = np.array(gv_diffs)
        tv_nashs = np.array(tv_nashs)

        print(
            f"  {k:>3}  {np.mean(ratios):>10.4f}±{np.std(ratios):.3f}"
            f"  {np.mean(gv_diffs):>8.4f}"
            f"  {np.mean(tv_nashs):>10.4f}"
            f"  {len(ratios):>8}"
        )

    # Bootstrap CIs for exploit ratios at key ranks
    boot_data = {}
    for k in [1, 2, 3, 5, 10]:
        ratios = []
        for r in results:
            rk = r["rank_results"].get(str(k))
            if rk and rk.get("status") != "lp_failed":
                ratios.append(rk["exploit_ratio"])
        if ratios:
            boot_data[f"exploit_ratio_k{k}"] = np.array(ratios)

    boot = bootstrap_all(boot_data, n_resamples=10_000, seed=42)

    print(f"\nBootstrap CIs for exploit ratio (R_k / R_full):")
    for k in [1, 2, 3, 5, 10]:
        key = f"exploit_ratio_k{k}"
        if key in boot:
            b = boot[key]
            print(
                f"  k={k}: {b['mean']:.4f} [{b['mean_ci_lo']:.4f}, {b['mean_ci_hi']:.4f}]"
            )

    # Save
    output = {
        "n_matchups": n,
        "ks_tested": ks,
        "per_matchup": results,
        "aggregate": {
            "bootstrap": boot,
            "summary": {},
        },
    }

    for k in ks:
        ratios = []
        gv_diffs = []
        tv_nashs = []
        for r in results:
            rk = r["rank_results"].get(str(k))
            if rk and rk.get("status") != "lp_failed":
                ratios.append(rk["exploit_ratio"])
                gv_diffs.append(abs(rk["game_value_diff"]))
                tv_nashs.append(rk["tv_nash_avg"])
        if ratios:
            output["aggregate"]["summary"][str(k)] = {
                "mean_exploit_ratio": float(np.mean(ratios)),
                "std_exploit_ratio": float(np.std(ratios)),
                "mean_gv_diff": float(np.mean(gv_diffs)),
                "mean_tv_nash": float(np.mean(tv_nashs)),
                "n_valid": len(ratios),
            }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
