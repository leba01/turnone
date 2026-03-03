"""Reward weight sensitivity: does BC-vs-BC ≈ Nash hold across reward specs?

Sweeps w_ko ∈ {1, 3, 5} (holding w_hp=1.0, w_field=0.5 fixed) and checks
that the core findings are robust to reward weight choices.

Usage:
    python scripts/reward_sensitivity.py \
        --bc_ckpt runs/bc_001/best.pt \
        --dyn_ckpt runs/dyn_001/best.pt \
        --test_split data/assembled/test.jsonl \
        --vocab_path runs/bc_001/vocab.json \
        --n_matchups 200 \
        --out_dir results/reward_sensitivity/
"""

from __future__ import annotations

import argparse
import json
import time
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
    exploitability_from_nash,
    bc_strategy_from_logits,
    compute_strategy_values,
)
from turnone.eval.bootstrap import bootstrap_all


def _build_bc_strategies(
    bc_model: BCPolicy,
    dataset: Turn1Dataset,
    indices: np.ndarray,
    device: torch.device,
) -> list[dict]:
    """Phase 0: build BC strategies + cache state dicts (reward-independent).

    Returns list of dicts with keys: idx, state, field_before_np,
    actions_p1, actions_p2, bc_p1, bc_p2.
    """
    matchups = []
    skipped = 0

    for idx in tqdm(indices, desc="Phase 0: BC strategies"):
        example = dataset[idx]

        mask_a_np = example["mask_a"].numpy()
        mask_b_np = example["mask_b"].numpy()
        opp_mask_a_np = example["opp_mask_a"].numpy()
        opp_mask_b_np = example["opp_mask_b"].numpy()

        # Strategic masks for enumeration
        strat_mask_a_np = example["strategic_mask_a"].numpy()
        strat_mask_b_np = example["strategic_mask_b"].numpy()
        opp_strat_mask_a_np = example["opp_strategic_mask_a"].numpy()
        opp_strat_mask_b_np = example["opp_strategic_mask_b"].numpy()

        actions_p1 = enumerate_joint_actions(
            strat_mask_a_np, strat_mask_b_np, include_tera=True
        )
        actions_p2 = enumerate_joint_actions(
            opp_strat_mask_a_np, opp_strat_mask_b_np, include_tera=True
        )

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

        matchups.append(
            {
                "idx": int(idx),
                "state": state,
                "field_before_np": field_before_np,
                "actions_p1": actions_p1,
                "actions_p2": actions_p2,
                "bc_p1": bc_p1,
                "bc_p2": bc_p2,
            }
        )

    if skipped > 0:
        print(f"  Skipped {skipped} matchups (empty action space)")
    print(f"  {len(matchups)} matchups prepared")
    return matchups


def _evaluate_with_weights(
    dyn_model: DynamicsModel,
    matchups: list[dict],
    reward_weights: dict[str, float],
    device: torch.device,
) -> dict:
    """Build payoff matrices with given weights, solve Nash, aggregate."""
    label = f"w_ko={reward_weights['w_ko']}"
    details = []

    for m in tqdm(matchups, desc=f"Payoff+Nash ({label})"):
        R = build_payoff_matrix(
            dyn_model,
            m["state"],
            m["actions_p1"],
            m["actions_p2"],
            m["field_before_np"],
            device,
            reward_weights=reward_weights,
        )

        try:
            nash_p1, nash_p2, game_value = solve_nash_lp(R)
        except ValueError:
            continue

        bc_exploit = exploitability_from_nash(m["bc_p1"], R, game_value, player=1)
        triangle = compute_strategy_values(m["bc_p1"], m["bc_p2"], R, game_value)

        details.append(
            {
                "idx": m["idx"],
                "game_value": float(game_value),
                "bc_exploitability": float(bc_exploit),
                **triangle,
            }
        )

    if len(details) == 0:
        return {"n_matchups": 0}

    exploits = np.array([d["bc_exploitability"] for d in details])
    game_vals = np.array([d["game_value"] for d in details])
    bc_vs_bc = np.array([d["bc_vs_bc"] for d in details])
    bc_worst = np.array([d["bc_worst_case"] for d in details])
    br_to_bc = np.array([d["best_response_to_bc"] for d in details])
    gap = bc_vs_bc - game_vals

    boot = bootstrap_all(
        {
            "bc_exploitability": exploits,
            "game_value": game_vals,
            "bc_vs_bc": bc_vs_bc,
            "bc_worst_case": bc_worst,
            "best_response_to_bc": br_to_bc,
            "bc_vs_bc_minus_nash": gap,
        },
        n_resamples=10_000,
        ci=0.95,
        seed=42,
    )

    return {
        "reward_weights": reward_weights,
        "n_matchups": len(details),
        "bc_exploitability_mean": boot["bc_exploitability"]["mean"],
        "bc_exploitability_mean_ci": [
            boot["bc_exploitability"]["mean_ci_lo"],
            boot["bc_exploitability"]["mean_ci_hi"],
        ],
        "game_value_mean": boot["game_value"]["mean"],
        "game_value_mean_ci": [
            boot["game_value"]["mean_ci_lo"],
            boot["game_value"]["mean_ci_hi"],
        ],
        "bc_vs_bc_mean": boot["bc_vs_bc"]["mean"],
        "bc_vs_bc_mean_ci": [
            boot["bc_vs_bc"]["mean_ci_lo"],
            boot["bc_vs_bc"]["mean_ci_hi"],
        ],
        "bc_vs_bc_minus_nash_mean": boot["bc_vs_bc_minus_nash"]["mean"],
        "bc_vs_bc_minus_nash_mean_ci": [
            boot["bc_vs_bc_minus_nash"]["mean_ci_lo"],
            boot["bc_vs_bc_minus_nash"]["mean_ci_hi"],
        ],
        "bc_worst_case_mean": boot["bc_worst_case"]["mean"],
        "best_response_to_bc_mean": boot["best_response_to_bc"]["mean"],
        "bootstrap": boot,
    }


def main():
    parser = argparse.ArgumentParser(description="Reward weight sensitivity sweep")
    parser.add_argument("--bc_ckpt", required=True)
    parser.add_argument("--dyn_ckpt", required=True)
    parser.add_argument("--test_split", required=True)
    parser.add_argument("--vocab_path", required=True)
    parser.add_argument("--n_matchups", type=int, default=500)
    parser.add_argument("--w_ko_values", type=float, nargs="+", default=[1.0, 3.0, 5.0])
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

    # Phase 0: build BC strategies (once, reward-independent)
    rng = np.random.RandomState(args.seed)
    indices = rng.choice(
        len(dataset), size=min(args.n_matchups, len(dataset)), replace=False
    )
    matchups = _build_bc_strategies(bc_model, dataset, indices, device)

    # Sweep w_ko values
    all_results = {}
    for w_ko in args.w_ko_values:
        rw = {"w_hp": 1.0, "w_ko": w_ko, "w_field": 0.5}
        print(f"\n=== w_ko = {w_ko} ===")
        t0 = time.time()

        result = _evaluate_with_weights(dyn_model, matchups, rw, device)

        print(
            f"  BC exploitability (mean): {result['bc_exploitability_mean']:.4f}  "
            f"CI {result['bc_exploitability_mean_ci']}"
        )
        print(
            f"  Nash value (mean):        {result['game_value_mean']:.4f}  "
            f"CI {result['game_value_mean_ci']}"
        )
        print(
            f"  BC-vs-BC (mean):          {result['bc_vs_bc_mean']:.4f}  "
            f"CI {result['bc_vs_bc_mean_ci']}"
        )
        print(
            f"  BC-vs-BC - Nash (mean):   {result['bc_vs_bc_minus_nash_mean']:.4f}  "
            f"CI {result['bc_vs_bc_minus_nash_mean_ci']}"
        )
        print(f"  Time: {time.time() - t0:.1f}s")

        all_results[f"w_ko_{w_ko}"] = result

    # Summary table
    print("\n\n=== SUMMARY ===")
    print(
        f"{'w_ko':>6}  {'exploit':>10}  {'nash_val':>10}  {'bc_vs_bc':>10}  {'gap':>10}"
    )
    print("-" * 55)
    for w_ko in args.w_ko_values:
        r = all_results[f"w_ko_{w_ko}"]
        print(
            f"{w_ko:>6.1f}  {r['bc_exploitability_mean']:>10.4f}  "
            f"{r['game_value_mean']:>10.4f}  {r['bc_vs_bc_mean']:>10.4f}  "
            f"{r['bc_vs_bc_minus_nash_mean']:>10.4f}"
        )

    with open(out_dir / "reward_sensitivity.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    main()
