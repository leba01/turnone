"""Phase 2 Experiment 2: QRE (Quantal Response Equilibrium) path analysis.

Computes QRE at multiple rationality levels (lambda) and measures distance
to BC and Nash strategies. Tests whether BC ≈ QRE at moderate lambda.

Usage:
    python scripts/phase2_qre.py \
        --cache results/phase2/cache.pkl \
        --out results/phase2/qre.json \
        --n_matchups 200 \
        --n_workers 8
"""

from __future__ import annotations

import argparse
import json
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

from turnone.game.exploitability import exploitability_from_nash
from turnone.eval.bootstrap import bootstrap_all


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    x = np.asarray(x, dtype=np.float64)
    x_max = x.max()
    e = np.exp(x - x_max)
    return e / e.sum()


def _tv(p: np.ndarray, q: np.ndarray) -> float:
    """Total variation distance."""
    return 0.5 * float(np.abs(p - q).sum())


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q), handling zeros. Returns inf if q=0 where p>0."""
    mask = p > 1e-15
    q_safe = np.maximum(q, 1e-15)
    return float(np.sum(p[mask] * np.log(p[mask] / q_safe[mask])))


def compute_qre(
    R: np.ndarray,
    lam: float,
    max_iter: int = 2000,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, int, bool]:
    """Compute QRE via simultaneous softmax best-response iteration.

    Args:
        R: (n1, n2) payoff matrix (P1's reward).
        lam: rationality parameter (higher = more rational, inf = Nash).
        max_iter: maximum iterations.
        tol: convergence tolerance (sum of TV changes).

    Returns:
        (x, y, iters, converged) — QRE strategies, iterations, convergence flag.
    """
    n1, n2 = R.shape
    x = np.ones(n1, dtype=np.float64) / n1  # uniform init
    y = np.ones(n2, dtype=np.float64) / n2

    for t in range(max_iter):
        x_new = _softmax(lam * (R @ y))
        y_new = _softmax(-lam * (x @ R))  # negate: P2 minimizes

        change = _tv(x_new, x) + _tv(y_new, y)
        x, y = x_new, y_new

        if change < tol:
            return x, y, t + 1, True

    return x, y, max_iter, False


def analyze_matchup(args: tuple) -> dict:
    """QRE analysis for one matchup across all lambda values."""
    idx, R, bc_p1, bc_p2, nash_p1, nash_p2, game_value, lambdas = args

    n1, n2 = R.shape
    results_by_lam = {}

    for lam in lambdas:
        x_qre, y_qre, iters, converged = compute_qre(R, lam)

        # QRE game value
        qre_value = float(x_qre @ R @ y_qre)

        # Distance to BC
        tv_bc_p1 = _tv(x_qre, bc_p1)
        tv_bc_p2 = _tv(y_qre, bc_p2)
        kl_bc_p1 = _kl(bc_p1, x_qre)  # KL(BC || QRE)
        kl_bc_p2 = _kl(bc_p2, y_qre)

        # Distance to Nash
        tv_nash_p1 = _tv(x_qre, nash_p1)
        tv_nash_p2 = _tv(y_qre, nash_p2)

        # QRE exploitability
        exploit_p1 = exploitability_from_nash(x_qre, R, game_value, player=1)
        exploit_p2 = exploitability_from_nash(y_qre, R, game_value, player=2)

        # QRE entropy
        entropy_p1 = -float(np.sum(x_qre[x_qre > 1e-15] * np.log(x_qre[x_qre > 1e-15])))
        entropy_p2 = -float(np.sum(y_qre[y_qre > 1e-15] * np.log(y_qre[y_qre > 1e-15])))

        results_by_lam[str(lam)] = {
            "lambda": lam,
            "iters": iters,
            "converged": converged,
            "qre_value": qre_value,
            "tv_bc_p1": tv_bc_p1,
            "tv_bc_p2": tv_bc_p2,
            "tv_bc_mean": (tv_bc_p1 + tv_bc_p2) / 2,
            "kl_bc_p1": kl_bc_p1,
            "kl_bc_p2": kl_bc_p2,
            "tv_nash_p1": tv_nash_p1,
            "tv_nash_p2": tv_nash_p2,
            "tv_nash_mean": (tv_nash_p1 + tv_nash_p2) / 2,
            "exploit_p1": float(exploit_p1),
            "exploit_p2": float(exploit_p2),
            "exploit_total": float(exploit_p1 + exploit_p2),
            "entropy_p1": entropy_p1,
            "entropy_p2": entropy_p2,
        }

    # Best-fit lambda: minimize mean TV(QRE, BC), only among converged results
    converged_results = [r for r in results_by_lam.values() if r["converged"]]
    if converged_results:
        best_lam = min(converged_results, key=lambda r: r["tv_bc_mean"])["lambda"]
    else:
        # Fallback: use lowest lambda (always converges)
        best_lam = min(results_by_lam.values(), key=lambda r: r["lambda"])["lambda"]

    # BC exploitability for comparison
    bc_exploit_p1 = exploitability_from_nash(bc_p1, R, game_value, player=1)
    bc_exploit_p2 = exploitability_from_nash(bc_p2, R, game_value, player=2)

    return {
        "idx": idx,
        "n1": n1,
        "n2": n2,
        "game_value": game_value,
        "bc_exploit_total": float(bc_exploit_p1 + bc_exploit_p2),
        "best_fit_lambda": best_lam,
        "best_fit_tv": results_by_lam[str(best_lam)]["tv_bc_mean"],
        "best_fit_exploit": results_by_lam[str(best_lam)]["exploit_total"],
        "best_fit_qre_value": results_by_lam[str(best_lam)]["qre_value"],
        "by_lambda": results_by_lam,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 2 Exp 2: QRE path analysis")
    parser.add_argument("--cache", required=True, help="Path to cache.pkl")
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument(
        "--n_matchups", type=int, default=500, help="Number of matchups to analyze"
    )
    parser.add_argument("--n_workers", type=int, default=8)
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=[0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 25, 50, 100],
    )
    args = parser.parse_args()

    with open(args.cache, "rb") as f:
        matchups = pickle.load(f)
    print(f"Loaded {len(matchups)} matchups from cache")

    # Use first n_matchups (they're already randomly sampled)
    matchups = matchups[: args.n_matchups]
    print(f"Using {len(matchups)} matchups")

    # Build work items
    work = [
        (
            m["idx"],
            m["R"],
            m["bc_p1"],
            m["bc_p2"],
            m["nash_p1"],
            m["nash_p2"],
            m["game_value"],
            args.lambdas,
        )
        for m in matchups
    ]

    # Run in parallel
    results = []
    if args.n_workers > 1 and len(work) > 1:
        with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
            futures = {executor.submit(analyze_matchup, w): w[0] for w in work}
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="QRE analysis"
            ):
                results.append(future.result())
    else:
        for w in tqdm(work, desc="QRE analysis"):
            results.append(analyze_matchup(w))

    # Aggregate: QRE path curves
    print(
        f"\n{'lambda':>8}  {'TV→BC':>8}  {'TV→Nash':>8}  {'QRE val':>8}  "
        f"{'exploit':>8}  {'entropy':>8}  {'conv%':>6}"
    )
    print("-" * 70)

    agg_by_lambda = {}
    for lam in args.lambdas:
        lam_key = str(lam)
        converged = np.array([r["by_lambda"][lam_key]["converged"] for r in results])
        tv_bc = np.array([r["by_lambda"][lam_key]["tv_bc_mean"] for r in results])
        tv_nash = np.array([r["by_lambda"][lam_key]["tv_nash_mean"] for r in results])
        qre_val = np.array([r["by_lambda"][lam_key]["qre_value"] for r in results])
        exploit = np.array([r["by_lambda"][lam_key]["exploit_total"] for r in results])
        entropy_p1 = np.array([r["by_lambda"][lam_key]["entropy_p1"] for r in results])
        conv_rate = float(converged.mean())

        agg_by_lambda[lam_key] = {
            "lambda": lam,
            "convergence_rate": conv_rate,
            "tv_bc_mean": float(tv_bc.mean()),
            "tv_bc_median": float(np.median(tv_bc)),
            "tv_nash_mean": float(tv_nash.mean()),
            "tv_nash_median": float(np.median(tv_nash)),
            "qre_value_mean": float(qre_val.mean()),
            "exploit_mean": float(exploit.mean()),
            "exploit_median": float(np.median(exploit)),
            "entropy_p1_mean": float(entropy_p1.mean()),
        }

        flag = "" if conv_rate > 0.95 else " *"
        print(
            f"{lam:>8.2f}  {tv_bc.mean():>8.3f}  {tv_nash.mean():>8.3f}  "
            f"{qre_val.mean():>8.3f}  {exploit.mean():>8.3f}  "
            f"{entropy_p1.mean():>8.2f}  {conv_rate * 100:>5.1f}{flag}"
        )

    print("(* = <95% convergence — results unreliable at this lambda)")

    # Best-fit lambda distribution
    best_lambdas = np.array([r["best_fit_lambda"] for r in results])
    best_fit_tvs = np.array([r["best_fit_tv"] for r in results])
    best_fit_exploits = np.array([r["best_fit_exploit"] for r in results])
    best_fit_values = np.array([r["best_fit_qre_value"] for r in results])
    game_values = np.array([r["game_value"] for r in results])
    bc_exploits = np.array([r["bc_exploit_total"] for r in results])

    print(f"\nBest-fit lambda distribution:")
    for lam in args.lambdas:
        frac = float((best_lambdas == lam).mean())
        if frac > 0:
            print(f"  λ={lam}: {frac * 100:.1f}%")

    print(f"\nAt best-fit λ*:")
    print(f"  TV(QRE, BC) mean: {best_fit_tvs.mean():.3f}")
    print(f"  QRE exploitability mean: {best_fit_exploits.mean():.3f}")
    print(f"  BC exploitability mean: {bc_exploits.mean():.3f}")
    print(f"  QRE value mean: {best_fit_values.mean():.4f}")
    print(f"  Nash value mean: {game_values.mean():.4f}")
    print(
        f"  |QRE value - V*| mean: {np.abs(best_fit_values - game_values).mean():.4f}"
    )

    # Bootstrap CIs for key metrics
    boot_data = {
        "best_fit_lambda": best_lambdas,
        "best_fit_tv": best_fit_tvs,
        "best_fit_exploit": best_fit_exploits,
        "value_gap_at_best_fit": np.abs(best_fit_values - game_values),
    }
    boot_results = bootstrap_all(boot_data, n_resamples=10_000, seed=42)

    # Convergence summary
    conv_summary = {}
    for lam in args.lambdas:
        lam_key = str(lam)
        conv_summary[lam_key] = agg_by_lambda[lam_key]["convergence_rate"]

    # Save
    output = {
        "n_matchups": len(results),
        "lambdas": args.lambdas,
        "convergence_rates": conv_summary,
        "qre_path": agg_by_lambda,
        "best_fit": {
            "lambda_distribution": {
                str(lam): float((best_lambdas == lam).mean()) for lam in args.lambdas
            },
            "tv_mean": float(best_fit_tvs.mean()),
            "exploit_mean": float(best_fit_exploits.mean()),
            "bc_exploit_mean": float(bc_exploits.mean()),
            "value_gap_mean": float(np.abs(best_fit_values - game_values).mean()),
            "bootstrap": boot_results,
        },
        "per_matchup": [
            {k: v for k, v in r.items() if k != "by_lambda"} for r in results
        ],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
