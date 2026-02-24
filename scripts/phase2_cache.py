"""Phase 2 cache builder: payoff matrices + BC/Nash strategies for 500 matchups.

Builds once on GPU, saves to pickle. All Phase 2 experiments read from cache.

Usage:
    python scripts/phase2_cache.py \
        --bc_ckpt runs/bc_001/best.pt \
        --dyn_ckpt runs/dyn_001/best.pt \
        --test_split data/assembled/test.jsonl \
        --vocab_path runs/bc_001/vocab.json \
        --n_matchups 500 \
        --out results/phase2/cache.pkl
"""

from __future__ import annotations

import argparse
import pickle
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
from turnone.game.exploitability import bc_strategy_from_logits


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

    for idx in tqdm(indices, desc="Building payoff matrices + BC + Nash"):
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
    print(f"  {len(matchups)} matchups cached")
    return matchups


def main():
    parser = argparse.ArgumentParser(description="Phase 2: build and cache payoff matrices")
    parser.add_argument("--bc_ckpt", required=True)
    parser.add_argument("--dyn_ckpt", required=True)
    parser.add_argument("--test_split", required=True)
    parser.add_argument("--vocab_path", required=True)
    parser.add_argument("--n_matchups", type=int, default=500)
    parser.add_argument("--out", required=True, help="Output pickle path")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load models
    print("Loading models...")
    bc_model = BCPolicy.from_checkpoint(args.bc_ckpt, device)
    dyn_model = DynamicsModel.from_checkpoint(args.dyn_ckpt, device)
    vocab = Vocab.load(args.vocab_path)
    dataset = Turn1Dataset(args.test_split, vocab)
    print(f"Test examples: {len(dataset):,}")

    # Sample matchups (same seed=42, same 500 indices as all other scripts)
    rng = np.random.RandomState(args.seed)
    indices = rng.choice(len(dataset), size=min(args.n_matchups, len(dataset)), replace=False)

    t0 = time.time()
    matchups = _build_matchups(bc_model, dyn_model, dataset, indices, device)
    elapsed = time.time() - t0

    # Save cache
    with open(out_path, "wb") as f:
        pickle.dump(matchups, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Summary
    sizes = [m["R"].shape for m in matchups]
    n1s = [s[0] for s in sizes]
    n2s = [s[1] for s in sizes]
    print(f"\nCache saved to {out_path}")
    print(f"  {len(matchups)} matchups, {elapsed:.1f}s")
    print(f"  Matrix sizes: P1 {np.mean(n1s):.0f}±{np.std(n1s):.0f}, "
          f"P2 {np.mean(n2s):.0f}±{np.std(n2s):.0f}")
    print(f"  File size: {out_path.stat().st_size / 1e6:.1f} MB")

    # Cross-check: print bc_vs_bc and game_value stats
    bc_vs_bc = [float(m["bc_p1"] @ m["R"] @ m["bc_p2"]) for m in matchups]
    game_vals = [m["game_value"] for m in matchups]
    print(f"  bc_vs_bc mean: {np.mean(bc_vs_bc):.4f}")
    print(f"  game_value mean: {np.mean(game_vals):.4f}")


if __name__ == "__main__":
    main()
