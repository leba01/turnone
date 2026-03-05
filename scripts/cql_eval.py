"""Evaluate CQL policy: exploitability, cross-play, and TV distances.

Compares BC (imitation) vs CQL (offline RL) vs Nash (game-theoretic optimum)
on a sample of test matchups.

Usage:
    python scripts/cql_eval.py \
        --cql_ckpt runs/cql_001/best.pt \
        --bc_ckpt runs/bc_001/best.pt \
        --dyn_ckpt runs/dyn_001/best.pt \
        --test_split data/assembled/test.jsonl \
        --vocab_path runs/bc_001/vocab.json \
        --n_matchups 500 \
        --out_dir results/cql_eval/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from turnone.data.dataset import Turn1Dataset, Vocab
from turnone.models.bc_policy import BCPolicy
from turnone.models.cql_qvalue import QLearner
from turnone.models.dynamics import DynamicsModel
from turnone.game.payoff import enumerate_joint_actions, build_payoff_matrix
from turnone.game.nash import solve_nash_lp
from turnone.game.exploitability import (
    exploitability_from_nash,
    bc_strategy_from_logits,
)


def cql_strategy_from_q(
    q_a: np.ndarray,
    q_b: np.ndarray,
    q_tera: np.ndarray,
    valid_actions: list[tuple[int, int, int]],
    tau: float = 1.0,
) -> np.ndarray:
    """Convert factored Q-values to a probability vector over joint actions.

    Args:
        q_a: (16,) Q-values for lead A.
        q_b: (16,) Q-values for lead B.
        q_tera: (3,) Q-values for tera.
        valid_actions: list of (slot_a, slot_b, tera_flag) tuples.
        tau: softmax temperature.

    Returns:
        (len(valid_actions),) probability vector.
    """
    if not valid_actions:
        return np.zeros(0, dtype=np.float64)

    q_joint = np.zeros(len(valid_actions), dtype=np.float64)
    for i, (sa, sb, tf) in enumerate(valid_actions):
        q_joint[i] = q_a[sa] + q_b[sb] + q_tera[tf]

    # Stable softmax
    q_joint = q_joint / tau
    q_max = np.max(q_joint)
    if not np.isfinite(q_max):
        q_max = 0.0
    e = np.exp(q_joint - q_max)
    return e / e.sum()


def tv_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Total variation distance between two probability vectors."""
    return 0.5 * float(np.sum(np.abs(p - q)))


def main():
    parser = argparse.ArgumentParser(description="Evaluate CQL vs BC vs Nash")
    parser.add_argument("--cql_ckpt", required=True)
    parser.add_argument("--bc_ckpt", required=True)
    parser.add_argument("--dyn_ckpt", required=True)
    parser.add_argument("--test_split", required=True)
    parser.add_argument("--vocab_path", required=True)
    parser.add_argument("--n_matchups", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load models
    print("Loading models...")
    cql_model = QLearner.from_checkpoint(args.cql_ckpt, device)
    bc_model = BCPolicy.from_checkpoint(args.bc_ckpt, device)
    dyn_model = DynamicsModel.from_checkpoint(args.dyn_ckpt, device)

    # Load data
    vocab = Vocab.load(args.vocab_path)
    dataset = Turn1Dataset(
        args.test_split, vocab,
        require_both_actions=True,
        canonicalize_targets=True,
    )

    # Sample matchups
    rng = np.random.RandomState(args.seed)
    n = min(args.n_matchups, len(dataset))
    indices = rng.choice(len(dataset), size=n, replace=False)
    indices.sort()

    print(f"Evaluating {n} matchups...")
    results = []
    skipped = 0

    for idx in tqdm(indices, desc="Evaluating"):
        example = dataset[int(idx)]

        # Enumerate valid actions
        strat_mask_a = example["strategic_mask_a"].numpy()
        strat_mask_b = example["strategic_mask_b"].numpy()
        opp_strat_mask_a = example["opp_strategic_mask_a"].numpy()
        opp_strat_mask_b = example["opp_strategic_mask_b"].numpy()

        actions_p1 = enumerate_joint_actions(strat_mask_a, strat_mask_b)
        actions_p2 = enumerate_joint_actions(opp_strat_mask_a, opp_strat_mask_b)

        if len(actions_p1) == 0 or len(actions_p2) == 0:
            skipped += 1
            continue

        # Build payoff matrix
        state = {k: example[k] for k in
                 ("team_a", "team_b", "lead_a", "lead_b", "field_state")}
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

        # BC strategy (P1)
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

        # CQL strategy (P1)
        with torch.no_grad():
            cql_out = cql_model(
                example["team_a"].unsqueeze(0).to(device),
                example["team_b"].unsqueeze(0).to(device),
                example["lead_a"].unsqueeze(0).to(device),
                example["lead_b"].unsqueeze(0).to(device),
                example["field_state"].unsqueeze(0).to(device),
            )
        cql_p1 = cql_strategy_from_q(
            cql_out["q_a"][0].cpu().numpy(),
            cql_out["q_b"][0].cpu().numpy(),
            cql_out["q_tera"][0].cpu().numpy(),
            actions_p1,
            tau=args.temperature,
        )

        # Exploitability
        bc_exploit = exploitability_from_nash(bc_p1, R, game_value, player=1)
        cql_exploit = exploitability_from_nash(cql_p1, R, game_value, player=1)

        # Cross-play values
        bc_vs_bc = float(bc_p1 @ R @ nash_p2)
        cql_vs_cql = float(cql_p1 @ R @ nash_p2)  # vs Nash P2
        cql_vs_bc_p2 = float(cql_p1 @ R @ nash_p2)

        # TV distances
        tv_bc_nash = tv_distance(bc_p1, nash_p1)
        tv_cql_nash = tv_distance(cql_p1, nash_p1)
        tv_cql_bc = tv_distance(cql_p1, bc_p1)

        results.append({
            "idx": int(idx),
            "n_actions_p1": len(actions_p1),
            "n_actions_p2": len(actions_p2),
            "game_value": game_value,
            "bc_exploit": bc_exploit,
            "cql_exploit": cql_exploit,
            "bc_worst_case": float(np.min(bc_p1 @ R)),
            "cql_worst_case": float(np.min(cql_p1 @ R)),
            "tv_bc_nash": tv_bc_nash,
            "tv_cql_nash": tv_cql_nash,
            "tv_cql_bc": tv_cql_bc,
        })

    if skipped:
        print(f"Skipped {skipped} matchups (empty actions or LP failure)")

    # Aggregate
    if not results:
        print("No valid results!")
        return

    agg = {}
    for key in ["bc_exploit", "cql_exploit", "tv_bc_nash", "tv_cql_nash", "tv_cql_bc",
                 "bc_worst_case", "cql_worst_case", "game_value"]:
        vals = [r[key] for r in results]
        agg[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "median": float(np.median(vals)),
            "ci_lo": float(np.percentile(vals, 2.5)),
            "ci_hi": float(np.percentile(vals, 97.5)),
        }

    # Save results
    output = {
        "n_matchups": len(results),
        "n_skipped": skipped,
        "temperature": args.temperature,
        "aggregate": agg,
        "per_matchup": results,
    }
    out_path = out_dir / "cql_eval.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print comparison table
    print(f"\n{'':>25}  {'BC':>10}  {'CQL':>10}  {'Nash':>10}")
    print("-" * 60)
    bc_e = agg["bc_exploit"]["mean"]
    cql_e = agg["cql_exploit"]["mean"]
    print(f"{'Exploitability':>25}  {bc_e:10.4f}  {cql_e:10.4f}  {'0.0000':>10}")
    tv_bn = agg["tv_bc_nash"]["mean"]
    tv_cn = agg["tv_cql_nash"]["mean"]
    print(f"{'TV(policy, Nash)':>25}  {tv_bn:10.4f}  {tv_cn:10.4f}  {'0.0000':>10}")
    tv_cb = agg["tv_cql_bc"]["mean"]
    print(f"{'TV(CQL, BC)':>25}  {'--':>10}  {tv_cb:10.4f}  {'--':>10}")
    bc_w = agg["bc_worst_case"]["mean"]
    cql_w = agg["cql_worst_case"]["mean"]
    gv = agg["game_value"]["mean"]
    print(f"{'Worst-case value':>25}  {bc_w:10.4f}  {cql_w:10.4f}  {gv:10.4f}")


if __name__ == "__main__":
    main()
