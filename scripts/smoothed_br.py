"""Smoothed best-response convergence from BC initialization.

Tests whether the metagame acts like a learning process: starting from BC
strategies, smoothed best-response dynamics should converge toward Nash.

Usage:
    python scripts/smoothed_br.py \
        --bc_ckpt runs/bc_001/best.pt \
        --dyn_ckpt runs/dyn_001/best.pt \
        --test_split data/assembled/test.jsonl \
        --vocab_path runs/bc_001/vocab.json \
        --n_matchups 200 \
        --out_dir results/smoothed_br/
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from turnone.data.dataset import Turn1Dataset, Vocab
from turnone.models.bc_policy import BCPolicy
from turnone.models.dynamics import DynamicsModel
from turnone.game.payoff import enumerate_joint_actions, build_payoff_matrix
from turnone.game.nash import solve_nash_lp
from turnone.game.exploitability import (
    exploitability_from_nash, bc_strategy_from_logits,
)


# ---------------------------------------------------------------------------
# Phase 0 (GPU): Build payoff matrices + BC strategies + solve Nash
# ---------------------------------------------------------------------------

def _build_matchups(
    bc_model: BCPolicy,
    dyn_model: DynamicsModel,
    dataset: Turn1Dataset,
    indices: np.ndarray,
    device: torch.device,
) -> list[dict]:
    """Build payoff matrices, BC strategies, and solve Nash for sampled matchups."""
    matchups = []
    skipped = 0

    for idx in tqdm(indices, desc="Phase 0: payoff + BC + Nash"):
        example = dataset[idx]

        strat_mask_a_np = example["strategic_mask_a"].numpy()
        strat_mask_b_np = example["strategic_mask_b"].numpy()
        opp_strat_mask_a_np = example["opp_strategic_mask_a"].numpy()
        opp_strat_mask_b_np = example["opp_strategic_mask_b"].numpy()

        actions_p1 = enumerate_joint_actions(strat_mask_a_np, strat_mask_b_np, include_tera=True)
        actions_p2 = enumerate_joint_actions(opp_strat_mask_a_np, opp_strat_mask_b_np, include_tera=True)

        if len(actions_p1) == 0 or len(actions_p2) == 0:
            skipped += 1
            continue

        state = {
            "team_a": example["team_a"],
            "team_b": example["team_b"],
            "lead_a": example["lead_a"],
            "lead_b": example["lead_b"],
            "field_state": example["field_state"],
        }
        field_before_np = example["field_state"].numpy()

        R = build_payoff_matrix(
            dyn_model, state, actions_p1, actions_p2,
            field_before_np, device,
        )

        # Solve Nash
        try:
            nash_p1, nash_p2, game_value = solve_nash_lp(R)
        except ValueError:
            skipped += 1
            continue

        # P1's BC strategy
        with torch.no_grad():
            bc_out = bc_model(
                example["team_a"].unsqueeze(0).to(device),
                example["team_b"].unsqueeze(0).to(device),
                example["lead_a"].unsqueeze(0).to(device),
                example["lead_b"].unsqueeze(0).to(device),
                example["field_state"].unsqueeze(0).to(device),
                example["mask_a"].unsqueeze(0).to(device),
                example["mask_b"].unsqueeze(0).to(device),
            )
        bc_p1 = bc_strategy_from_logits(
            bc_out["logits_a"][0].cpu().numpy(),
            bc_out["logits_b"][0].cpu().numpy(),
            bc_out["logits_tera"][0].cpu().numpy(),
            actions_p1,
        )

        # P2's BC strategy (perspective swap)
        field_p2 = example["field_state"].clone()
        field_p2[3], field_p2[4] = field_p2[4].item(), field_p2[3].item()

        with torch.no_grad():
            bc_out_p2 = bc_model(
                example["team_b"].unsqueeze(0).to(device),
                example["team_a"].unsqueeze(0).to(device),
                example["lead_b"].unsqueeze(0).to(device),
                example["lead_a"].unsqueeze(0).to(device),
                field_p2.unsqueeze(0).to(device),
                example["opp_mask_a"].unsqueeze(0).to(device),
                example["opp_mask_b"].unsqueeze(0).to(device),
            )
        bc_p2 = bc_strategy_from_logits(
            bc_out_p2["logits_a"][0].cpu().numpy(),
            bc_out_p2["logits_b"][0].cpu().numpy(),
            bc_out_p2["logits_tera"][0].cpu().numpy(),
            actions_p2,
        )

        matchups.append({
            "idx": int(idx),
            "R": R,
            "bc_p1": bc_p1,
            "bc_p2": bc_p2,
            "nash_p1": nash_p1,
            "nash_p2": nash_p2,
            "game_value": float(game_value),
        })

    if skipped > 0:
        print(f"  Skipped {skipped} matchups (empty action space or LP failure)")
    print(f"  {len(matchups)} matchups prepared")
    return matchups


# ---------------------------------------------------------------------------
# Phase 1 (CPU): Smoothed best-response dynamics
# ---------------------------------------------------------------------------

def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    x = np.asarray(x, dtype=np.float64)
    x_max = x.max()
    e = np.exp(x - x_max)
    return e / e.sum()


def _run_smoothed_br(args: tuple) -> dict:
    """Run smoothed BR for one matchup across all (beta, eta) combos.

    Tracks both instantaneous strategies and time-averaged strategies.
    In zero-sum games, the time-average of smoothed BR converges to Nash
    even when instantaneous strategies converge to QRE (the quantal response
    equilibrium at temperature beta).
    """
    idx, R, bc_p1, bc_p2, nash_p1, nash_p2, game_value, betas, etas, n_iters = args

    results_by_param = {}

    for beta in betas:
        for eta in etas:
            key = f"beta={beta}_eta={eta}"

            x = bc_p1.copy()  # P1 strategy (row player)
            y = bc_p2.copy()  # P2 strategy (column player)

            # Running sums for time-averaged strategies
            x_sum = bc_p1.copy()
            y_sum = bc_p2.copy()

            trajectory = []
            for t in range(n_iters + 1):
                # Time-averaged strategies
                x_avg = x_sum / (t + 1)
                y_avg = y_sum / (t + 1)

                # Instantaneous exploitability
                exploit_p1 = exploitability_from_nash(x, R, game_value, player=1)
                exploit_p2 = exploitability_from_nash(y, R, game_value, player=2)
                total_exploit = exploit_p1 + exploit_p2

                # Time-averaged exploitability (the one that should → 0)
                avg_exploit_p1 = exploitability_from_nash(x_avg, R, game_value, player=1)
                avg_exploit_p2 = exploitability_from_nash(y_avg, R, game_value, player=2)
                avg_total_exploit = avg_exploit_p1 + avg_exploit_p2

                tv_to_nash_p1 = 0.5 * float(np.abs(x_avg - nash_p1).sum())
                tv_to_nash_p2 = 0.5 * float(np.abs(y_avg - nash_p2).sum())

                value = float(x @ R @ y)

                trajectory.append({
                    "t": t,
                    "exploit": float(total_exploit),
                    "avg_exploit": float(avg_total_exploit),
                    "avg_tv_p1": tv_to_nash_p1,
                    "avg_tv_p2": tv_to_nash_p2,
                    "value": value,
                })

                if t < n_iters:
                    # Simultaneous update
                    # P1: best-responds to P2's strategy y
                    payoffs_p1 = R @ y           # (n1,) expected payoff per P1 action
                    sbr_p1 = _softmax(payoffs_p1 / beta)
                    x_new = (1 - eta) * x + eta * sbr_p1

                    # P2: best-responds to P1's strategy x (minimizes P1's payoff)
                    payoffs_p2 = x @ R           # (n2,) P1's expected payoff per P2 action
                    sbr_p2 = _softmax(-payoffs_p2 / beta)  # Negate: P2 minimizes
                    y_new = (1 - eta) * y + eta * sbr_p2

                    x = x_new
                    y = y_new
                    x_sum += x
                    y_sum += y

            results_by_param[key] = {
                "beta": beta,
                "eta": eta,
                "trajectory": trajectory,
                "final_exploit": trajectory[-1]["exploit"],
                "final_avg_exploit": trajectory[-1]["avg_exploit"],
                "final_avg_tv_p1": trajectory[-1]["avg_tv_p1"],
                "final_avg_tv_p2": trajectory[-1]["avg_tv_p2"],
            }

    return {
        "idx": idx,
        "game_value": game_value,
        "n_actions_p1": R.shape[0],
        "n_actions_p2": R.shape[1],
        "initial_exploit_p1": float(exploitability_from_nash(bc_p1, R, game_value, player=1)),
        "initial_exploit_p2": float(exploitability_from_nash(bc_p2, R, game_value, player=2)),
        "params": results_by_param,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _aggregate_trajectories(
    all_results: list[dict],
    betas: list[float],
    etas: list[float],
    n_iters: int,
) -> dict:
    """Aggregate per-matchup trajectories into mean/median curves."""
    n = len(all_results)
    agg = {}

    for beta in betas:
        for eta in etas:
            key = f"beta={beta}_eta={eta}"

            # Collect per-iteration arrays
            exploit_matrix = np.zeros((n, n_iters + 1))
            avg_exploit_matrix = np.zeros((n, n_iters + 1))
            avg_tv_p1_matrix = np.zeros((n, n_iters + 1))
            avg_tv_p2_matrix = np.zeros((n, n_iters + 1))
            value_matrix = np.zeros((n, n_iters + 1))

            for i, r in enumerate(all_results):
                traj = r["params"][key]["trajectory"]
                for t in range(n_iters + 1):
                    exploit_matrix[i, t] = traj[t]["exploit"]
                    avg_exploit_matrix[i, t] = traj[t]["avg_exploit"]
                    avg_tv_p1_matrix[i, t] = traj[t]["avg_tv_p1"]
                    avg_tv_p2_matrix[i, t] = traj[t]["avg_tv_p2"]
                    value_matrix[i, t] = traj[t]["value"]

            final_avg_exploits = avg_exploit_matrix[:, -1]

            agg[key] = {
                "beta": beta,
                "eta": eta,
                "n_matchups": n,
                "mean_trajectory": {
                    "exploit": exploit_matrix.mean(axis=0).tolist(),
                    "avg_exploit": avg_exploit_matrix.mean(axis=0).tolist(),
                    "avg_tv_p1": avg_tv_p1_matrix.mean(axis=0).tolist(),
                    "avg_tv_p2": avg_tv_p2_matrix.mean(axis=0).tolist(),
                    "value": value_matrix.mean(axis=0).tolist(),
                },
                "median_trajectory": {
                    "avg_exploit": np.median(avg_exploit_matrix, axis=0).tolist(),
                },
                "p25_trajectory": {
                    "avg_exploit": np.percentile(avg_exploit_matrix, 25, axis=0).tolist(),
                },
                "p75_trajectory": {
                    "avg_exploit": np.percentile(avg_exploit_matrix, 75, axis=0).tolist(),
                },
                "final_avg_exploit_mean": float(final_avg_exploits.mean()),
                "final_avg_exploit_median": float(np.median(final_avg_exploits)),
                "final_avg_exploit_max": float(final_avg_exploits.max()),
                "converged_fraction": float((final_avg_exploits < 0.1).mean()),
            }

    return agg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Smoothed best-response convergence")
    parser.add_argument("--bc_ckpt", required=True)
    parser.add_argument("--dyn_ckpt", required=True)
    parser.add_argument("--test_split", required=True)
    parser.add_argument("--vocab_path", required=True)
    parser.add_argument("--n_matchups", type=int, default=200)
    parser.add_argument("--betas", type=float, nargs="+", default=[0.01, 0.05, 0.1, 0.5, 1.0])
    parser.add_argument("--etas", type=float, nargs="+", default=[0.1, 0.3])
    parser.add_argument("--n_iters", type=int, default=500)
    parser.add_argument("--n_workers", type=int, default=8)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load models
    print("Loading models...")
    bc_model = BCPolicy.from_checkpoint(args.bc_ckpt, device)
    dyn_model = DynamicsModel.from_checkpoint(args.dyn_ckpt, device)
    vocab = Vocab.load(args.vocab_path)
    dataset = Turn1Dataset(args.test_split, vocab)
    print(f"Test examples: {len(dataset):,}")

    # Phase 0: GPU — build payoff matrices + BC + Nash
    rng = np.random.RandomState(args.seed)
    indices = rng.choice(len(dataset), size=min(args.n_matchups, len(dataset)), replace=False)

    t0 = time.time()
    matchups = _build_matchups(bc_model, dyn_model, dataset, indices, device)
    print(f"Phase 0 time: {time.time() - t0:.1f}s")

    # Phase 1: CPU parallel — smoothed BR dynamics
    print(f"\nPhase 1: Smoothed BR ({len(matchups)} matchups × "
          f"{len(args.betas)*len(args.etas)} param combos × {args.n_iters} iters)...")
    t1 = time.time()

    matchup_args = [
        (m["idx"], m["R"], m["bc_p1"], m["bc_p2"],
         m["nash_p1"], m["nash_p2"], m["game_value"],
         args.betas, args.etas, args.n_iters)
        for m in matchups
    ]

    all_results = []
    if args.n_workers > 1 and len(matchup_args) > 1:
        with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
            futures = {
                executor.submit(_run_smoothed_br, ma): ma[0]
                for ma in matchup_args
            }
            for future in tqdm(
                as_completed(futures), total=len(futures),
                desc="Phase 1: Smoothed BR (CPU parallel)",
            ):
                all_results.append(future.result())
    else:
        for ma in tqdm(matchup_args, desc="Phase 1: Smoothed BR"):
            all_results.append(_run_smoothed_br(ma))

    print(f"Phase 1 time: {time.time() - t1:.1f}s")

    # Phase 2: Aggregate
    print("\nPhase 2: Aggregation...")
    agg = _aggregate_trajectories(all_results, args.betas, args.etas, args.n_iters)

    # Print summary
    print(f"\n{'='*70}")
    print(f"SMOOTHED BR CONVERGENCE ({len(all_results)} matchups)")
    print(f"{'='*70}")

    init_exploit = np.mean([r["initial_exploit_p1"] + r["initial_exploit_p2"]
                            for r in all_results])
    print(f"Initial exploitability (BC): {init_exploit:.4f}")

    print(f"\n{'beta':>6}  {'eta':>5}  {'QRE exploit':>12}  {'avg exploit':>12}  "
          f"{'converged%':>10}")
    print("-" * 55)

    for beta in args.betas:
        for eta in args.etas:
            key = f"beta={beta}_eta={eta}"
            a = agg[key]
            # QRE exploit = instantaneous final, avg exploit = time-averaged final
            qre = a["mean_trajectory"]["exploit"][-1]
            print(f"{beta:>6.2f}  {eta:>5.1f}  {qre:>12.4f}  "
                  f"{a['final_avg_exploit_mean']:>12.4f}  "
                  f"{a['converged_fraction']*100:>9.1f}%")

    # Save aggregated results (without per-matchup raw trajectories for size)
    with open(out_dir / "smoothed_br.json", "w") as f:
        json.dump({
            "n_matchups": len(all_results),
            "betas": args.betas,
            "etas": args.etas,
            "n_iters": args.n_iters,
            "initial_exploit_mean": float(init_exploit),
            "aggregated": agg,
        }, f, indent=2)

    # Save per-matchup details (larger file, for detailed analysis)
    # Strip full trajectories, keep only summaries per matchup
    matchup_summaries = []
    for r in all_results:
        summary = {
            "idx": r["idx"],
            "game_value": r["game_value"],
            "n_actions_p1": r["n_actions_p1"],
            "n_actions_p2": r["n_actions_p2"],
            "initial_exploit_p1": r["initial_exploit_p1"],
            "initial_exploit_p2": r["initial_exploit_p2"],
        }
        for key, pdata in r["params"].items():
            summary[f"{key}_final_exploit"] = pdata["final_exploit"]
            summary[f"{key}_final_avg_exploit"] = pdata["final_avg_exploit"]
            summary[f"{key}_final_avg_tv_p1"] = pdata["final_avg_tv_p1"]
            summary[f"{key}_final_avg_tv_p2"] = pdata["final_avg_tv_p2"]
        matchup_summaries.append(summary)

    with open(out_dir / "matchup_summaries.jsonl", "w") as f:
        for s in matchup_summaries:
            f.write(json.dumps(s) + "\n")

    print(f"\nTotal time: {time.time() - t0:.1f}s")
    print(f"Results saved to {out_dir}/")


if __name__ == "__main__":
    main()
