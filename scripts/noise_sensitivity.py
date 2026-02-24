"""Noise injection experiment: how robust is Nash to dynamics model error?

Takes matchup payoff matrices, adds calibrated Gaussian noise, re-solves Nash,
and measures total variation distance + exploitability change.

Usage:
    python scripts/noise_sensitivity.py \
        --matchup_details results/matchup_details.jsonl \
        --bc_ckpt runs/bc_001/best.pt \
        --dyn_ckpt runs/dyn_001/best.pt \
        --test_split data/assembled/test.jsonl \
        --vocab_path runs/bc_001/vocab.json \
        --reward_mae 0.5 \
        --n_matchups 50 \
        --out_dir results/noise_sensitivity/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from turnone.game.nash import solve_nash_lp
from turnone.game.exploitability import exploitability_from_nash


def noise_sensitivity_single(
    R: np.ndarray,
    bc_strategy: np.ndarray,
    noise_std: float,
    n_trials: int = 20,
    seed: int = 0,
) -> dict[str, float]:
    """Run noise injection on one payoff matrix.

    Args:
        R: (n1, n2) clean payoff matrix.
        bc_strategy: (n1,) BC strategy for P1.
        noise_std: std dev of Gaussian noise to add to each entry.
        n_trials: number of noisy re-solves.
        seed: random seed.

    Returns:
        Dict with tv_distance_mean/std, exploit_change_mean/std.
    """
    rng = np.random.RandomState(seed)

    # Solve clean game
    try:
        clean_p1, clean_p2, clean_value = solve_nash_lp(R)
    except ValueError:
        return None

    clean_exploit = exploitability_from_nash(bc_strategy, R, clean_value, player=1)

    tv_distances = []
    exploit_changes = []
    value_changes = []

    for _ in range(n_trials):
        noise = rng.randn(*R.shape).astype(np.float32) * noise_std
        R_noisy = R + noise

        try:
            noisy_p1, noisy_p2, noisy_value = solve_nash_lp(R_noisy)
        except ValueError:
            continue

        # Total variation distance between clean and noisy Nash (P1)
        # TV = 0.5 * sum |p - q|
        # Need to handle different-sized strategies if action spaces differ (they don't here)
        tv = 0.5 * np.abs(clean_p1 - noisy_p1).sum()
        tv_distances.append(float(tv))

        # Exploitability change (using CLEAN R to evaluate noisy strategy)
        noisy_exploit = exploitability_from_nash(noisy_p1, R, clean_value, player=1)
        exploit_changes.append(float(noisy_exploit - 0.0))  # Nash should be 0, noisy may not be

        value_changes.append(float(abs(noisy_value - clean_value)))

    if len(tv_distances) == 0:
        return None

    tv_arr = np.array(tv_distances)
    ec_arr = np.array(exploit_changes)
    vc_arr = np.array(value_changes)

    return {
        "n_trials": len(tv_distances),
        "clean_value": float(clean_value),
        "clean_exploit": float(clean_exploit),
        "tv_distance_mean": float(tv_arr.mean()),
        "tv_distance_std": float(tv_arr.std()),
        "noisy_exploit_mean": float(ec_arr.mean()),
        "noisy_exploit_std": float(ec_arr.std()),
        "value_change_mean": float(vc_arr.mean()),
        "value_change_std": float(vc_arr.std()),
    }


def main():
    parser = argparse.ArgumentParser(description="Noise sensitivity experiment")
    parser.add_argument("--bc_ckpt", required=True)
    parser.add_argument("--dyn_ckpt", required=True)
    parser.add_argument("--test_split", required=True)
    parser.add_argument("--vocab_path", required=True)
    parser.add_argument("--reward_mae", type=float, default=None,
                        help="Measured reward MAE (noise calibration). If not given, reads from dynamics_metrics.json")
    parser.add_argument("--metrics_json", default="results/dynamics_metrics.json",
                        help="Path to dynamics_metrics.json for auto-calibration")
    parser.add_argument("--n_matchups", type=int, default=50)
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--noise_scales", type=float, nargs="+",
                        default=[0.25, 0.5, 1.0, 1.5, 2.0],
                        help="Multipliers for reward_mae to sweep noise levels")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import torch
    from turnone.data.dataset import Turn1Dataset, Vocab
    from turnone.models.bc_policy import BCPolicy
    from turnone.models.dynamics import DynamicsModel
    from turnone.game.payoff import enumerate_joint_actions, build_payoff_matrix
    from turnone.game.exploitability import bc_strategy_from_logits

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Get reward MAE for calibration
    if args.reward_mae is not None:
        reward_mae = args.reward_mae
    else:
        with open(args.metrics_json) as f:
            metrics = json.load(f)
        reward_mae = metrics["reward_mae"]
    print(f"Calibration reward MAE: {reward_mae:.4f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    bc_model = BCPolicy.from_checkpoint(args.bc_ckpt, device)
    dyn_model = DynamicsModel.from_checkpoint(args.dyn_ckpt, device)
    vocab = Vocab.load(args.vocab_path)
    dataset = Turn1Dataset(args.test_split, vocab)

    # Sample matchups
    rng = np.random.RandomState(args.seed)
    indices = rng.choice(len(dataset), size=min(args.n_matchups, len(dataset)), replace=False)

    # Build payoff matrices + BC strategies
    print(f"Building {len(indices)} payoff matrices...")
    matchups = []
    for idx in tqdm(indices, desc="Building payoff matrices"):
        example = dataset[idx]
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
            continue

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

        bc_strategy = bc_strategy_from_logits(
            bc_out["logits_a"][0].cpu().numpy(),
            bc_out["logits_b"][0].cpu().numpy(),
            bc_out["logits_tera"][0].cpu().numpy(),
            actions_p1,
        )

        matchups.append((int(idx), R, bc_strategy))

    print(f"Built {len(matchups)} payoff matrices")

    # Sweep noise levels
    all_results = {}
    for scale in args.noise_scales:
        noise_std = reward_mae * scale
        print(f"\n--- Noise scale {scale}x (std={noise_std:.4f}) ---")

        per_matchup = []
        for idx, R, bc_strat in tqdm(matchups, desc=f"Scale {scale}x"):
            result = noise_sensitivity_single(
                R, bc_strat, noise_std,
                n_trials=args.n_trials, seed=args.seed + idx,
            )
            if result is not None:
                result["idx"] = idx
                per_matchup.append(result)

        if len(per_matchup) == 0:
            continue

        tv_means = np.array([r["tv_distance_mean"] for r in per_matchup])
        exploit_means = np.array([r["noisy_exploit_mean"] for r in per_matchup])
        value_means = np.array([r["value_change_mean"] for r in per_matchup])

        summary = {
            "noise_scale": scale,
            "noise_std": float(noise_std),
            "n_matchups": len(per_matchup),
            "tv_distance_mean": float(tv_means.mean()),
            "tv_distance_std": float(tv_means.std()),
            "noisy_exploit_mean": float(exploit_means.mean()),
            "noisy_exploit_std": float(exploit_means.std()),
            "value_change_mean": float(value_means.mean()),
            "value_change_std": float(value_means.std()),
        }
        all_results[f"scale_{scale}"] = summary

        print(f"  TV distance: {summary['tv_distance_mean']:.4f} +/- {summary['tv_distance_std']:.4f}")
        print(f"  Noisy exploit: {summary['noisy_exploit_mean']:.4f} +/- {summary['noisy_exploit_std']:.4f}")
        print(f"  Value change: {summary['value_change_mean']:.4f} +/- {summary['value_change_std']:.4f}")

    with open(out_dir / "noise_sensitivity.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    main()
