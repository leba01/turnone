"""End-to-end evaluation: dynamics metrics + exploitability on test matchups.

Usage:
    python scripts/evaluate.py \
        --bc_ckpt runs/bc_001/best.pt \
        --dyn_ckpt runs/dyn_001/best.pt \
        --test_split data/assembled/test.jsonl \
        --n_matchups 500 \
        --out_dir results/
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
from turnone.models.dynamics import DynamicsModel, remap_actions
from turnone.rl.reward import compute_reward_from_dynamics
from turnone.game.nash import solve_nash_lp
from turnone.game.exploitability import (
    exploitability_from_nash, bc_strategy_from_logits,
    bc_strategy_autoregressive, compute_strategy_values,
)
from turnone.game.payoff import enumerate_joint_actions, build_payoff_matrix
from turnone.eval.dynamics_metrics import compute_dynamics_metrics, compute_reward_error
from turnone.eval.bootstrap import bootstrap_all


def evaluate_dynamics(
    dyn_model: DynamicsModel,
    dataset: Turn1Dataset,
    device: torch.device,
    batch_size: int = 512,
) -> tuple[dict[str, float], dict[str, float]]:
    """Evaluate dynamics model on a dataset. Returns (metrics, reward_error) dicts."""
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_hp_pred, all_hp_true = [], []
    all_ko_logits, all_ko_true = [], []
    all_field_pred, all_field_true = [], []
    all_field_before = []

    for batch in tqdm(loader, desc="Dynamics eval", leave=False):
        team_a = batch["team_a"].to(device)
        team_b = batch["team_b"].to(device)
        lead_a = batch["lead_a"].to(device)
        lead_b = batch["lead_b"].to(device)
        field_state = batch["field_state"].to(device)
        tera_flag = batch["tera_label"].to(device)
        opp_tera_flag = batch["opp_tera_label"].to(device)

        # Remap actions
        action_a = remap_actions(batch["action_a"].to(device))
        action_b = remap_actions(batch["action_b"].to(device))
        opp_action_a = remap_actions(batch["opp_action_a"].to(device))
        opp_action_b = remap_actions(batch["opp_action_b"].to(device))

        with torch.no_grad():
            hp_pred, ko_logits, field_logits = dyn_model(
                team_a, team_b, lead_a, lead_b, field_state,
                action_a, action_b, tera_flag,
                opp_action_a, opp_action_b, opp_tera_flag,
            )
            field_pred = dyn_model.predict_field_state(field_logits)

        all_hp_pred.append(hp_pred.float().cpu().numpy())
        all_hp_true.append(batch["hp_delta"].numpy())
        all_ko_logits.append(ko_logits.float().cpu().numpy())
        all_ko_true.append(batch["ko_flags"].numpy())
        all_field_pred.append(field_pred.float().cpu().numpy())
        all_field_true.append(batch["field_after"].numpy())
        all_field_before.append(batch["field_state"].numpy())

    hp_pred = np.concatenate(all_hp_pred)
    hp_true = np.concatenate(all_hp_true)
    ko_logits = np.concatenate(all_ko_logits)
    ko_true = np.concatenate(all_ko_true)
    field_pred = np.concatenate(all_field_pred)
    field_true = np.concatenate(all_field_true)
    field_before = np.concatenate(all_field_before)

    dyn_metrics = compute_dynamics_metrics(
        hp_pred=hp_pred, hp_true=hp_true,
        ko_logits=ko_logits, ko_true=ko_true,
        field_pred=field_pred, field_true=field_true,
    )

    reward_err = compute_reward_error(
        hp_pred=hp_pred, hp_true=hp_true,
        ko_logits=ko_logits, ko_true=ko_true,
        field_pred=field_pred, field_true=field_true,
        field_before=field_before,
    )

    return dyn_metrics, reward_err


def _solve_single_matchup(args: tuple) -> dict | None:
    """Solve Nash + compute exploitability + triangle values for one matchup.

    Args is a tuple of (idx, R, bc_p1, bc_p2) to be picklable for ProcessPoolExecutor.
    """
    idx, R, bc_p1, bc_p2 = args
    try:
        nash_p1, nash_p2, game_value = solve_nash_lp(R)
    except ValueError:
        return None

    bc_exploit = exploitability_from_nash(bc_p1, R, game_value, player=1)
    nash_exploit = exploitability_from_nash(nash_p1, R, game_value, player=1)

    # Safety-exploitation triangle (P1)
    triangle = compute_strategy_values(bc_p1, bc_p2, R, game_value)

    # Nash support size (number of actions with prob > 1e-4)
    nash_support_p1 = int((nash_p1 > 1e-4).sum())
    nash_support_p2 = int((nash_p2 > 1e-4).sum())

    return {
        "idx": int(idx),
        "n_actions_p1": R.shape[0],
        "n_actions_p2": R.shape[1],
        "game_value": float(game_value),
        "bc_exploitability": float(bc_exploit),
        "nash_exploitability": float(nash_exploit),
        "nash_support_p1": nash_support_p1,
        "nash_support_p2": nash_support_p2,
        **triangle,
    }


def evaluate_exploitability(
    bc_model: BCPolicy,
    dyn_model: DynamicsModel,
    dataset: Turn1Dataset,
    device: torch.device,
    n_matchups: int = 500,
    seed: int = 42,
    n_workers: int = 8,
    reward_weights: dict[str, float] | None = None,
    autoregressive: bool = False,
) -> dict:
    """Evaluate BC exploitability on sampled matchups.

    Two-phase pipeline:
      Phase 1 (GPU): Build all payoff matrices + BC strategies (both sides).
      Phase 2 (CPU parallel): Solve Nash LPs + compute exploitability + triangle.
    """
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(dataset), size=min(n_matchups, len(dataset)), replace=False)

    # ---- Phase 1: GPU — build payoff matrices + BC strategies ----
    matchup_data = []  # list of (idx, R, bc_p1, bc_p2)
    skipped = 0

    for idx in tqdm(indices, desc="Phase 1: payoff matrices (GPU)"):
        example = dataset[idx]

        # Get masks for both sides (BC masks for model forward pass)
        mask_a_np = example["mask_a"].numpy()
        mask_b_np = example["mask_b"].numpy()
        opp_mask_a_np = example["opp_mask_a"].numpy()
        opp_mask_b_np = example["opp_mask_b"].numpy()

        # Strategic masks for enumeration (no target=3 for single-target moves)
        strat_mask_a_np = example["strategic_mask_a"].numpy()
        strat_mask_b_np = example["strategic_mask_b"].numpy()
        opp_strat_mask_a_np = example["opp_strategic_mask_a"].numpy()
        opp_strat_mask_b_np = example["opp_strategic_mask_b"].numpy()

        # Enumerate valid actions using strategic masks
        actions_p1 = enumerate_joint_actions(strat_mask_a_np, strat_mask_b_np, include_tera=True)
        actions_p2 = enumerate_joint_actions(opp_strat_mask_a_np, opp_strat_mask_b_np, include_tera=True)

        if len(actions_p1) == 0 or len(actions_p2) == 0:
            skipped += 1
            continue

        # Build state dict for dynamics model
        state = {
            "team_a": example["team_a"],
            "team_b": example["team_b"],
            "lead_a": example["lead_a"],
            "lead_b": example["lead_b"],
            "field_state": example["field_state"],
        }
        field_before_np = example["field_state"].numpy()

        # Build payoff matrix (uses cached encoder internally)
        R = build_payoff_matrix(
            dyn_model, state, actions_p1, actions_p2,
            field_before_np, device,
            reward_weights=reward_weights,
        )

        # --- P1's BC strategy ---
        if autoregressive and bc_model.autoregressive:
            bc_p1 = bc_strategy_autoregressive(
                bc_model, example, actions_p1, device)
        else:
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

        # --- P2's BC strategy (perspective swap) ---
        # P2 sees: their team (our team_b) as team_a, our team as team_b
        # Swap tailwind indices (3, 4) in field state
        field_p2 = example["field_state"].clone()
        field_p2[3], field_p2[4] = field_p2[4].item(), field_p2[3].item()

        # Build P2 perspective example for autoregressive
        p2_example = {
            "team_a": example["team_b"],
            "team_b": example["team_a"],
            "lead_a": example["lead_b"],
            "lead_b": example["lead_a"],
            "field_state": field_p2,
            "mask_a": example["opp_mask_a"],
            "mask_b": example["opp_mask_b"],
        }

        if autoregressive and bc_model.autoregressive:
            bc_p2 = bc_strategy_autoregressive(
                bc_model, p2_example, actions_p2, device)
        else:
            with torch.no_grad():
                bc_out_p2 = bc_model(
                    p2_example["team_a"].unsqueeze(0).to(device),
                    p2_example["team_b"].unsqueeze(0).to(device),
                    p2_example["lead_a"].unsqueeze(0).to(device),
                    p2_example["lead_b"].unsqueeze(0).to(device),
                    p2_example["field_state"].unsqueeze(0).to(device),
                    p2_example["mask_a"].unsqueeze(0).to(device),
                    p2_example["mask_b"].unsqueeze(0).to(device),
                )
            bc_p2 = bc_strategy_from_logits(
                bc_out_p2["logits_a"][0].cpu().numpy(),
                bc_out_p2["logits_b"][0].cpu().numpy(),
                bc_out_p2["logits_tera"][0].cpu().numpy(),
                actions_p2,
            )

        matchup_data.append((int(idx), R, bc_p1, bc_p2))

    if skipped > 0:
        print(f"  Skipped {skipped} matchups (empty action space)")

    print(f"  Phase 1 done: {len(matchup_data)} payoff matrices built")

    # ---- Phase 2: CPU parallel — solve Nash LPs ----
    matchup_details = []

    if n_workers > 1 and len(matchup_data) > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_solve_single_matchup, md): md[0]
                for md in matchup_data
            }
            for future in tqdm(
                as_completed(futures), total=len(futures),
                desc="Phase 2: Nash LPs (CPU parallel)",
            ):
                result = future.result()
                if result is not None:
                    matchup_details.append(result)
    else:
        # Sequential fallback (tests, small runs)
        for md in tqdm(matchup_data, desc="Phase 2: Nash LPs"):
            result = _solve_single_matchup(md)
            if result is not None:
                matchup_details.append(result)

    # Summary statistics
    if len(matchup_details) == 0:
        return {"n_matchups": 0}, []

    exploitabilities = np.array([d["bc_exploitability"] for d in matchup_details])
    game_values = np.array([d["game_value"] for d in matchup_details])
    bc_worst = np.array([d["bc_worst_case"] for d in matchup_details])
    bc_vs_bc = np.array([d["bc_vs_bc"] for d in matchup_details])
    br_to_bc = np.array([d["best_response_to_bc"] for d in matchup_details])

    # Bootstrap CIs
    boot_data = {
        "bc_exploitability": exploitabilities,
        "game_value": game_values,
        "bc_worst_case": bc_worst,
        "bc_vs_bc": bc_vs_bc,
        "best_response_to_bc": br_to_bc,
        "stackelberg_gap": br_to_bc - game_values,
    }
    boot_results = bootstrap_all(boot_data, n_resamples=10_000, ci=0.95, seed=42)

    results = {
        "n_matchups": len(matchup_details),
        # Exploitability
        "bc_exploitability_mean": boot_results["bc_exploitability"]["mean"],
        "bc_exploitability_mean_ci": [
            boot_results["bc_exploitability"]["mean_ci_lo"],
            boot_results["bc_exploitability"]["mean_ci_hi"],
        ],
        "bc_exploitability_median": boot_results["bc_exploitability"]["median"],
        "bc_exploitability_median_ci": [
            boot_results["bc_exploitability"]["median_ci_lo"],
            boot_results["bc_exploitability"]["median_ci_hi"],
        ],
        "bc_exploitability_std": float(exploitabilities.std()),
        # Game value
        "game_value_mean": boot_results["game_value"]["mean"],
        "game_value_std": float(game_values.std()),
        # Triangle values
        "bc_worst_case_mean": boot_results["bc_worst_case"]["mean"],
        "bc_worst_case_mean_ci": [
            boot_results["bc_worst_case"]["mean_ci_lo"],
            boot_results["bc_worst_case"]["mean_ci_hi"],
        ],
        "bc_vs_bc_mean": boot_results["bc_vs_bc"]["mean"],
        "bc_vs_bc_mean_ci": [
            boot_results["bc_vs_bc"]["mean_ci_lo"],
            boot_results["bc_vs_bc"]["mean_ci_hi"],
        ],
        "best_response_to_bc_mean": boot_results["best_response_to_bc"]["mean"],
        "best_response_to_bc_mean_ci": [
            boot_results["best_response_to_bc"]["mean_ci_lo"],
            boot_results["best_response_to_bc"]["mean_ci_hi"],
        ],
        "stackelberg_gap_mean": boot_results["stackelberg_gap"]["mean"],
        "stackelberg_gap_mean_ci": [
            boot_results["stackelberg_gap"]["mean_ci_lo"],
            boot_results["stackelberg_gap"]["mean_ci_hi"],
        ],
        # Full bootstrap results
        "bootstrap": boot_results,
    }

    return results, matchup_details


def main():
    parser = argparse.ArgumentParser(description="End-to-end evaluation")
    parser.add_argument("--bc_ckpt", required=True, help="Path to BC checkpoint")
    parser.add_argument("--dyn_ckpt", required=True, help="Path to dynamics checkpoint")
    parser.add_argument("--test_split", required=True, help="Path to test JSONL")
    parser.add_argument("--vocab_path", required=True, help="Path to vocab.json")
    parser.add_argument("--n_matchups", type=int, default=500, help="Number of matchups")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--n_workers", type=int, default=8, help="CPU workers for Nash LP")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--autoregressive", action="store_true",
                        help="Use autoregressive BC strategy (P(b|a)) instead of independent")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load models
    print("Loading BC model...")
    bc_model = BCPolicy.from_checkpoint(args.bc_ckpt, device)
    print("Loading dynamics model...")
    dyn_model = DynamicsModel.from_checkpoint(args.dyn_ckpt, device)

    # Load dataset
    print("Loading test data...")
    vocab = Vocab.load(args.vocab_path)
    test_ds = Turn1Dataset(args.test_split, vocab)
    print(f"Test examples: {len(test_ds):,}")

    # 1. Dynamics metrics + reward error (P2)
    print("\n=== Dynamics Metrics ===")
    t0 = time.time()
    dyn_metrics, reward_err = evaluate_dynamics(dyn_model, test_ds, device)
    print(f"  HP MAE:  {dyn_metrics['hp_mae']:.4f}")
    print(f"  HP RMSE: {dyn_metrics['hp_rmse']:.4f}")
    print(f"  HP R²:   {dyn_metrics['hp_r2']:.4f}")
    print(f"  KO AUC:  {dyn_metrics['ko_auc']:.4f}")
    print(f"  KO Acc:  {dyn_metrics['ko_acc']:.4f}")
    print(f"  Weather Acc: {dyn_metrics['weather_acc']:.4f}  (not used in reward)")
    print(f"  Terrain Acc: {dyn_metrics['terrain_acc']:.4f}  (not used in reward)")
    print(f"  Binary Field Acc: {dyn_metrics['binary_field_acc']:.4f}")
    print(f"\n  Reward-space error:")
    print(f"    Reward MAE:  {reward_err['reward_mae']:.4f}")
    print(f"    Reward RMSE: {reward_err['reward_rmse']:.4f}")
    print(f"    Reward corr: {reward_err['reward_correlation']:.4f}")
    print(f"    Reward bias: {reward_err['reward_bias']:.4f}")
    print(f"  Time: {time.time() - t0:.1f}s")

    combined_metrics = {**dyn_metrics, **reward_err}
    with open(out_dir / "dynamics_metrics.json", "w") as f:
        json.dump(combined_metrics, f, indent=2)

    # 2. Exploitability + triangle + bootstrap (P1, P4)
    print(f"\n=== Exploitability ({args.n_matchups} matchups) ===")
    t0 = time.time()
    exploit_results, matchup_details = evaluate_exploitability(
        bc_model, dyn_model, test_ds, device,
        n_matchups=args.n_matchups, seed=args.seed,
        n_workers=args.n_workers,
        autoregressive=args.autoregressive,
    )
    print(f"  Matchups evaluated: {exploit_results['n_matchups']}")
    print(f"  BC exploitability (mean):   {exploit_results['bc_exploitability_mean']:.4f}  "
          f"95% CI [{exploit_results['bc_exploitability_mean_ci'][0]:.4f}, "
          f"{exploit_results['bc_exploitability_mean_ci'][1]:.4f}]")
    print(f"  BC exploitability (median): {exploit_results['bc_exploitability_median']:.4f}  "
          f"95% CI [{exploit_results['bc_exploitability_median_ci'][0]:.4f}, "
          f"{exploit_results['bc_exploitability_median_ci'][1]:.4f}]")
    print(f"  Game value (mean): {exploit_results['game_value_mean']:.4f}")

    print(f"\n  Safety-exploitation triangle (means):")
    print(f"    BC worst-case:       {exploit_results['bc_worst_case_mean']:.4f}  "
          f"CI [{exploit_results['bc_worst_case_mean_ci'][0]:.4f}, "
          f"{exploit_results['bc_worst_case_mean_ci'][1]:.4f}]")
    print(f"    Nash value:          {exploit_results['game_value_mean']:.4f}")
    print(f"    BC-vs-BC:            {exploit_results['bc_vs_bc_mean']:.4f}  "
          f"CI [{exploit_results['bc_vs_bc_mean_ci'][0]:.4f}, "
          f"{exploit_results['bc_vs_bc_mean_ci'][1]:.4f}]")
    print(f"    Best-resp-to-BC:     {exploit_results['best_response_to_bc_mean']:.4f}  "
          f"CI [{exploit_results['best_response_to_bc_mean_ci'][0]:.4f}, "
          f"{exploit_results['best_response_to_bc_mean_ci'][1]:.4f}]")
    print(f"    Stackelberg gap:     {exploit_results['stackelberg_gap_mean']:.4f}  "
          f"CI [{exploit_results['stackelberg_gap_mean_ci'][0]:.4f}, "
          f"{exploit_results['stackelberg_gap_mean_ci'][1]:.4f}]")
    print(f"  Time: {time.time() - t0:.1f}s")

    with open(out_dir / "exploitability_results.json", "w") as f:
        json.dump(exploit_results, f, indent=2)

    with open(out_dir / "matchup_details.jsonl", "w") as f:
        for detail in matchup_details:
            f.write(json.dumps(detail) + "\n")

    print(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    main()
