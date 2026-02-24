"""Case study deep-dives: show BC vs Nash distributions for individual matchups.

Picks matchups that illustrate different regimes and shows actual Pokemon names,
move names, BC probability distributions, and Nash distributions side by side.

Usage:
    python scripts/case_studies.py \
        --bc_ckpt runs/bc_001/best.pt \
        --dyn_ckpt runs/dyn_001/best.pt \
        --test_split data/assembled/test.jsonl \
        --vocab_path runs/bc_001/vocab.json \
        --matchup_details results/matchup_details.jsonl \
        --out_dir results/case_studies/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from turnone.data.dataset import Turn1Dataset, Vocab
from turnone.data.action_space import SLOTS_PER_MON
from turnone.models.bc_policy import BCPolicy
from turnone.models.dynamics import DynamicsModel
from turnone.game.nash import solve_nash_lp
from turnone.game.payoff import enumerate_joint_actions, build_payoff_matrix
from turnone.game.exploitability import bc_strategy_from_logits, exploitability_from_nash


def _reverse_vocab(vocab: Vocab) -> dict[str, dict[int, str]]:
    """Build idx->token reverse mappings."""
    rev = {}
    for field_type, tok2idx in vocab._tok2idx.items():
        rev[field_type] = {idx: tok for tok, idx in tok2idx.items()}
    return rev


def _decode_mon(team_tensor: torch.Tensor, mon_idx: int, rev: dict[str, dict[int, str]]) -> dict:
    """Decode a single mon from the (6, 8) team tensor."""
    row = team_tensor[mon_idx].tolist()
    return {
        "species": rev["species"].get(row[0], "???"),
        "item": rev["item"].get(row[1], "???"),
        "ability": rev["ability"].get(row[2], "???"),
        "tera_type": rev["tera_type"].get(row[3], "???"),
        "moves": [
            rev["move"].get(row[4], "???"),
            rev["move"].get(row[5], "???"),
            rev["move"].get(row[6], "???"),
            rev["move"].get(row[7], "???"),
        ],
    }


def _action_label(slot: int, mon: dict) -> str:
    """Convert a slot index (0-15) to a human-readable action label."""
    move_idx = slot // 4
    target_idx = slot % 4
    move_name = mon["moves"][move_idx] if move_idx < len(mon["moves"]) else "???"
    target_names = ["opp_A", "opp_B", "ally", "self"]
    target = target_names[target_idx]
    return f"{move_name} -> {target}"


def _joint_action_label(action: tuple[int, int, int], mon_a: dict, mon_b: dict) -> str:
    """Format a joint action (slot_a, slot_b, tera) as readable string."""
    sa, sb, tera = action
    a_str = _action_label(sa, mon_a)
    b_str = _action_label(sb, mon_b)
    tera_str = ["none", "tera_A", "tera_B"][tera]
    return f"[{a_str}] + [{b_str}] ({tera_str})"


def analyze_matchup(
    idx: int,
    dataset: Turn1Dataset,
    bc_model: BCPolicy,
    dyn_model: DynamicsModel,
    vocab: Vocab,
    device: torch.device,
    top_k: int = 5,
) -> dict:
    """Full analysis of a single matchup: BC vs Nash with Pokemon names."""
    rev = _reverse_vocab(vocab)
    example = dataset[idx]

    # Decode teams
    lead_a_indices = example["lead_a"].tolist()
    lead_b_indices = example["lead_b"].tolist()

    p1_lead_a = _decode_mon(example["team_a"], lead_a_indices[0], rev)
    p1_lead_b = _decode_mon(example["team_a"], lead_a_indices[1], rev)
    p2_lead_a = _decode_mon(example["team_b"], lead_b_indices[0], rev)
    p2_lead_b = _decode_mon(example["team_b"], lead_b_indices[1], rev)

    # Masks for BC forward pass
    mask_a_np = example["mask_a"].numpy()
    mask_b_np = example["mask_b"].numpy()

    # Strategic masks for enumeration
    strat_mask_a_np = example["strategic_mask_a"].numpy()
    strat_mask_b_np = example["strategic_mask_b"].numpy()
    opp_strat_mask_a_np = example["opp_strategic_mask_a"].numpy()
    opp_strat_mask_b_np = example["opp_strategic_mask_b"].numpy()

    actions_p1 = enumerate_joint_actions(strat_mask_a_np, strat_mask_b_np, include_tera=True)
    actions_p2 = enumerate_joint_actions(opp_strat_mask_a_np, opp_strat_mask_b_np, include_tera=True)

    if len(actions_p1) == 0 or len(actions_p2) == 0:
        return None

    # Build payoff matrix
    state = {
        "team_a": example["team_a"],
        "team_b": example["team_b"],
        "lead_a": example["lead_a"],
        "lead_b": example["lead_b"],
        "field_state": example["field_state"],
    }
    R = build_payoff_matrix(
        dyn_model, state, actions_p1, actions_p2,
        example["field_state"].numpy(), device,
    )

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

    # Nash
    try:
        nash_p1, nash_p2, game_value = solve_nash_lp(R)
    except ValueError:
        return None

    bc_exploit = exploitability_from_nash(bc_p1, R, game_value, player=1)

    # Nash support
    nash_support_mask = nash_p1 > 1e-4
    nash_support_size = int(nash_support_mask.sum())

    # Top-K actions for BC and Nash
    bc_top_k_idx = np.argsort(bc_p1)[::-1][:top_k]
    nash_top_k_idx = np.argsort(nash_p1)[::-1][:top_k]

    bc_top_actions = []
    for i in bc_top_k_idx:
        bc_top_actions.append({
            "action": _joint_action_label(actions_p1[i], p1_lead_a, p1_lead_b),
            "bc_prob": float(bc_p1[i]),
            "nash_prob": float(nash_p1[i]),
        })

    nash_top_actions = []
    for i in nash_top_k_idx:
        nash_top_actions.append({
            "action": _joint_action_label(actions_p1[i], p1_lead_a, p1_lead_b),
            "nash_prob": float(nash_p1[i]),
            "bc_prob": float(bc_p1[i]),
        })

    return {
        "dataset_idx": int(idx),
        "p1_leads": {
            "lead_a": f"{p1_lead_a['species']} ({p1_lead_a['ability']})",
            "lead_b": f"{p1_lead_b['species']} ({p1_lead_b['ability']})",
        },
        "p2_leads": {
            "lead_a": f"{p2_lead_a['species']} ({p2_lead_a['ability']})",
            "lead_b": f"{p2_lead_b['species']} ({p2_lead_b['ability']})",
        },
        "n_actions_p1": len(actions_p1),
        "n_actions_p2": len(actions_p2),
        "game_value": float(game_value),
        "bc_exploitability": float(bc_exploit),
        "nash_support_size": nash_support_size,
        "bc_top_actions": bc_top_actions,
        "nash_top_actions": nash_top_actions,
    }


def select_case_studies(matchup_details: list[dict], n_each: int = 1) -> dict[str, list[int]]:
    """Select matchups that illustrate different regimes.

    Returns dict mapping regime name to list of dataset indices.
    """
    if len(matchup_details) == 0:
        return {}

    by_exploit = sorted(matchup_details, key=lambda d: d["bc_exploitability"])
    by_support = sorted(matchup_details, key=lambda d: d.get("nash_support_p1", 0))

    selections = {}

    # Near-Nash: lowest exploitability
    selections["near_nash"] = [d["idx"] for d in by_exploit[:n_each]]

    # Most exploitable: highest exploitability
    selections["most_exploitable"] = [d["idx"] for d in by_exploit[-n_each:]]

    # Large Nash support
    selections["large_support"] = [d["idx"] for d in by_support[-n_each:]]

    # Small Nash support (but not trivial)
    non_trivial = [d for d in by_support if d.get("nash_support_p1", 0) >= 2]
    if non_trivial:
        selections["small_support"] = [d["idx"] for d in non_trivial[:n_each]]

    # Median exploitability
    mid = len(by_exploit) // 2
    selections["median_exploit"] = [d["idx"] for d in by_exploit[mid:mid + n_each]]

    return selections


def main():
    parser = argparse.ArgumentParser(description="Case study analysis")
    parser.add_argument("--bc_ckpt", required=True)
    parser.add_argument("--dyn_ckpt", required=True)
    parser.add_argument("--test_split", required=True)
    parser.add_argument("--vocab_path", required=True)
    parser.add_argument("--matchup_details", required=True,
                        help="Path to matchup_details.jsonl from evaluate.py")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    bc_model = BCPolicy.from_checkpoint(args.bc_ckpt, device)
    dyn_model = DynamicsModel.from_checkpoint(args.dyn_ckpt, device)
    vocab = Vocab.load(args.vocab_path)
    dataset = Turn1Dataset(args.test_split, vocab)

    # Load matchup details for selection
    matchup_details = []
    with open(args.matchup_details) as f:
        for line in f:
            matchup_details.append(json.loads(line))

    # Select case studies
    selections = select_case_studies(matchup_details, n_each=1)
    print("Selected case studies:")
    for regime, indices in selections.items():
        print(f"  {regime}: indices {indices}")

    # Analyze each
    all_studies = {}
    all_indices = set()
    for regime, indices in selections.items():
        all_indices.update(indices)

    for idx in tqdm(sorted(all_indices), desc="Analyzing matchups"):
        result = analyze_matchup(idx, dataset, bc_model, dyn_model, vocab, device, args.top_k)
        if result is None:
            continue

        # Find which regimes this index belongs to
        regimes = [r for r, idxs in selections.items() if idx in idxs]
        result["regimes"] = regimes
        all_studies[idx] = result

        # Print summary
        print(f"\n{'='*60}")
        print(f"Matchup {idx} ({', '.join(regimes)})")
        print(f"  P1: {result['p1_leads']['lead_a']} + {result['p1_leads']['lead_b']}")
        print(f"  P2: {result['p2_leads']['lead_a']} + {result['p2_leads']['lead_b']}")
        print(f"  Game value: {result['game_value']:.4f}")
        print(f"  BC exploitability: {result['bc_exploitability']:.4f}")
        print(f"  Nash support: {result['nash_support_size']} actions")
        print(f"  Action space: {result['n_actions_p1']} x {result['n_actions_p2']}")

        print(f"\n  BC top-{args.top_k} actions:")
        for a in result["bc_top_actions"]:
            print(f"    {a['bc_prob']:.3f} (Nash: {a['nash_prob']:.3f})  {a['action']}")

        print(f"\n  Nash top-{args.top_k} actions:")
        for a in result["nash_top_actions"]:
            print(f"    {a['nash_prob']:.3f} (BC: {a['bc_prob']:.3f})  {a['action']}")

    # Save
    with open(out_dir / "case_studies.json", "w") as f:
        json.dump({str(k): v for k, v in all_studies.items()}, f, indent=2)

    print(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    main()
