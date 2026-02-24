"""Value decomposition + strategy distance analysis.

Extension A: Cross-play values to diagnose WHY bc_vs_bc ≈ Nash.
Extension C: Strategy-space distances (TV, support overlap) to quantify the paradox.

Usage:
    python scripts/value_decomposition.py \
        --bc_ckpt runs/bc_001/best.pt \
        --dyn_ckpt runs/dyn_001/best.pt \
        --test_split data/assembled/test.jsonl \
        --vocab_path runs/bc_001/vocab.json \
        --n_matchups 500 \
        --out_dir results/value_decomposition/
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from tqdm import tqdm

from turnone.data.dataset import Turn1Dataset, Vocab
from turnone.models.bc_policy import BCPolicy
from turnone.models.dynamics import DynamicsModel
from turnone.game.payoff import enumerate_joint_actions, build_payoff_matrix
from turnone.game.nash import solve_nash_lp
from turnone.game.exploitability import (
    exploitability_from_nash, bc_strategy_from_logits,
)
from turnone.eval.bootstrap import bootstrap_all


# ---------------------------------------------------------------------------
# Phase 0 (GPU): Build payoff matrices + BC strategies
# ---------------------------------------------------------------------------

def _build_matchups(
    bc_model: BCPolicy,
    dyn_model: DynamicsModel,
    dataset: Turn1Dataset,
    indices: np.ndarray,
    device: torch.device,
) -> list[dict]:
    """Build payoff matrices and BC strategies for sampled matchups."""
    matchups = []
    skipped = 0

    for idx in tqdm(indices, desc="Phase 0: payoff matrices + BC (GPU)"):
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
        })

    if skipped > 0:
        print(f"  Skipped {skipped} matchups (empty action space)")
    print(f"  {len(matchups)} matchups prepared")
    return matchups


# ---------------------------------------------------------------------------
# Phase 1 (CPU): Nash + cross-play values + strategy distances
# ---------------------------------------------------------------------------

SUPPORT_THRESH = 1e-4


def _analyze_single_matchup(args: tuple) -> dict | None:
    """Solve Nash, compute cross-play values and strategy distances."""
    idx, R, bc_p1, bc_p2 = args

    try:
        nash_p1, nash_p2, game_value = solve_nash_lp(R)
    except ValueError:
        return None

    # --- Extension A: Value decomposition ---
    bc_vs_bc = float(bc_p1 @ R @ bc_p2)
    bc_p1_vs_nash_p2 = float(bc_p1 @ R @ nash_p2)
    nash_p1_vs_bc_p2 = float(nash_p1 @ R @ bc_p2)
    bc_worst_case = float(np.min(bc_p1 @ R))
    best_response_to_bc = float(np.max(R @ bc_p2))

    bc_exploit_p1 = exploitability_from_nash(bc_p1, R, game_value, player=1)
    bc_exploit_p2 = exploitability_from_nash(bc_p2, R, game_value, player=2)

    # Cross-play gaps (how far from V* each cross-play value is)
    cross_gap_p1 = bc_p1_vs_nash_p2 - game_value  # <0 if BC P1 loses vs Nash P2
    cross_gap_p2 = nash_p1_vs_bc_p2 - game_value  # >0 if Nash P1 exploits BC P2

    # --- Extension C: Strategy distances ---
    # Total variation
    tv_p1 = 0.5 * float(np.abs(bc_p1 - nash_p1).sum())
    tv_p2 = 0.5 * float(np.abs(bc_p2 - nash_p2).sum())

    # Support analysis
    nash_supp_p1 = nash_p1 > SUPPORT_THRESH
    nash_supp_p2 = nash_p2 > SUPPORT_THRESH
    bc_supp_p1 = bc_p1 > SUPPORT_THRESH
    bc_supp_p2 = bc_p2 > SUPPORT_THRESH

    nash_supp_size_p1 = int(nash_supp_p1.sum())
    nash_supp_size_p2 = int(nash_supp_p2.sum())
    bc_supp_size_p1 = int(bc_supp_p1.sum())
    bc_supp_size_p2 = int(bc_supp_p2.sum())

    # Support overlap: fraction of Nash support that BC also plays
    overlap_p1 = float((nash_supp_p1 & bc_supp_p1).sum()) / max(nash_supp_size_p1, 1)
    overlap_p2 = float((nash_supp_p2 & bc_supp_p2).sum()) / max(nash_supp_size_p2, 1)

    # BC mass on Nash support actions
    bc_mass_on_nash_p1 = float(bc_p1[nash_supp_p1].sum()) if nash_supp_size_p1 > 0 else 0.0
    bc_mass_on_nash_p2 = float(bc_p2[nash_supp_p2].sum()) if nash_supp_size_p2 > 0 else 0.0

    return {
        "idx": idx,
        "n_actions_p1": R.shape[0],
        "n_actions_p2": R.shape[1],
        # Value decomposition
        "game_value": float(game_value),
        "bc_vs_bc": bc_vs_bc,
        "bc_p1_vs_nash_p2": bc_p1_vs_nash_p2,
        "nash_p1_vs_bc_p2": nash_p1_vs_bc_p2,
        "bc_worst_case": bc_worst_case,
        "best_response_to_bc": best_response_to_bc,
        "bc_exploit_p1": float(bc_exploit_p1),
        "bc_exploit_p2": float(bc_exploit_p2),
        "cross_gap_p1": cross_gap_p1,
        "cross_gap_p2": cross_gap_p2,
        # Strategy distances
        "tv_p1": tv_p1,
        "tv_p2": tv_p2,
        "nash_supp_size_p1": nash_supp_size_p1,
        "nash_supp_size_p2": nash_supp_size_p2,
        "bc_supp_size_p1": bc_supp_size_p1,
        "bc_supp_size_p2": bc_supp_size_p2,
        "support_overlap_p1": overlap_p1,
        "support_overlap_p2": overlap_p2,
        "bc_mass_on_nash_p1": bc_mass_on_nash_p1,
        "bc_mass_on_nash_p2": bc_mass_on_nash_p2,
    }


# ---------------------------------------------------------------------------
# Phase 2: Aggregate + regression
# ---------------------------------------------------------------------------

def _aggregate(details: list[dict]) -> dict:
    """Aggregate per-matchup results with bootstrap CIs + regression."""
    n = len(details)

    # Extract arrays
    game_values = np.array([d["game_value"] for d in details])
    bc_vs_bc = np.array([d["bc_vs_bc"] for d in details])
    bc_p1_vs_nash_p2 = np.array([d["bc_p1_vs_nash_p2"] for d in details])
    nash_p1_vs_bc_p2 = np.array([d["nash_p1_vs_bc_p2"] for d in details])
    bc_worst = np.array([d["bc_worst_case"] for d in details])
    br_to_bc = np.array([d["best_response_to_bc"] for d in details])
    bc_exploit_p1 = np.array([d["bc_exploit_p1"] for d in details])
    bc_exploit_p2 = np.array([d["bc_exploit_p2"] for d in details])
    cross_gap_p1 = np.array([d["cross_gap_p1"] for d in details])
    cross_gap_p2 = np.array([d["cross_gap_p2"] for d in details])

    tv_p1 = np.array([d["tv_p1"] for d in details])
    tv_p2 = np.array([d["tv_p2"] for d in details])
    nash_supp_p1 = np.array([d["nash_supp_size_p1"] for d in details])
    nash_supp_p2 = np.array([d["nash_supp_size_p2"] for d in details])
    bc_supp_p1 = np.array([d["bc_supp_size_p1"] for d in details])
    bc_supp_p2 = np.array([d["bc_supp_size_p2"] for d in details])
    overlap_p1 = np.array([d["support_overlap_p1"] for d in details])
    overlap_p2 = np.array([d["support_overlap_p2"] for d in details])
    bc_mass_nash_p1 = np.array([d["bc_mass_on_nash_p1"] for d in details])
    bc_mass_nash_p2 = np.array([d["bc_mass_on_nash_p2"] for d in details])

    # Bootstrap CIs for key quantities
    boot = bootstrap_all({
        "game_value": game_values,
        "bc_vs_bc": bc_vs_bc,
        "bc_p1_vs_nash_p2": bc_p1_vs_nash_p2,
        "nash_p1_vs_bc_p2": nash_p1_vs_bc_p2,
        "bc_worst_case": bc_worst,
        "best_response_to_bc": br_to_bc,
        "bc_exploit_p1": bc_exploit_p1,
        "bc_exploit_p2": bc_exploit_p2,
        "cross_gap_p1": cross_gap_p1,
        "cross_gap_p2": cross_gap_p2,
        "bc_vs_bc_minus_nash": bc_vs_bc - game_values,
        "tv_p1": tv_p1,
        "tv_p2": tv_p2,
        "support_overlap_p1": overlap_p1,
        "support_overlap_p2": overlap_p2,
        "bc_mass_on_nash_p1": bc_mass_nash_p1,
        "bc_mass_on_nash_p2": bc_mass_nash_p2,
    }, n_resamples=10_000, ci=0.95, seed=42)

    # Regression: exploitability ~ TV distance
    tv_avg = (tv_p1 + tv_p2) / 2.0
    exploit_avg = (bc_exploit_p1 + bc_exploit_p2) / 2.0
    r_tv_exploit, p_tv_exploit = stats.pearsonr(tv_avg, exploit_avg)
    slope, intercept, _, _, _ = stats.linregress(tv_avg, exploit_avg)

    # Regression: exploitability ~ Nash support size
    supp_avg = (nash_supp_p1 + nash_supp_p2) / 2.0
    r_supp_exploit, p_supp_exploit = stats.pearsonr(supp_avg, exploit_avg)

    # Regression: exploitability ~ game size
    game_sizes = np.array([d["n_actions_p1"] * d["n_actions_p2"] for d in details], dtype=float)
    r_size_exploit, p_size_exploit = stats.pearsonr(game_sizes, exploit_avg)

    # Cross-gap analysis: are they opposite-signed (error cancellation)?
    n_same_sign = int(np.sum(cross_gap_p1 * cross_gap_p2 > 0))
    n_opposite_sign = int(np.sum(cross_gap_p1 * cross_gap_p2 < 0))
    n_zero = n - n_same_sign - n_opposite_sign

    # Absolute cross-play gaps
    abs_cross_gap_p1 = np.abs(cross_gap_p1)
    abs_cross_gap_p2 = np.abs(cross_gap_p2)

    return {
        "n_matchups": n,
        # --- Extension A: Value decomposition ---
        "value_decomposition": {
            "game_value": _ci_dict(boot, "game_value"),
            "bc_vs_bc": _ci_dict(boot, "bc_vs_bc"),
            "bc_p1_vs_nash_p2": _ci_dict(boot, "bc_p1_vs_nash_p2"),
            "nash_p1_vs_bc_p2": _ci_dict(boot, "nash_p1_vs_bc_p2"),
            "bc_worst_case": _ci_dict(boot, "bc_worst_case"),
            "best_response_to_bc": _ci_dict(boot, "best_response_to_bc"),
            "bc_exploit_p1": _ci_dict(boot, "bc_exploit_p1"),
            "bc_exploit_p2": _ci_dict(boot, "bc_exploit_p2"),
            "bc_vs_bc_minus_nash": _ci_dict(boot, "bc_vs_bc_minus_nash"),
        },
        "cross_play_gaps": {
            "cross_gap_p1": _ci_dict(boot, "cross_gap_p1"),
            "cross_gap_p2": _ci_dict(boot, "cross_gap_p2"),
            "abs_cross_gap_p1_mean": float(abs_cross_gap_p1.mean()),
            "abs_cross_gap_p2_mean": float(abs_cross_gap_p2.mean()),
            "n_same_sign": n_same_sign,
            "n_opposite_sign": n_opposite_sign,
            "n_zero": n_zero,
        },
        # --- Extension C: Strategy distances ---
        "strategy_distances": {
            "tv_p1": _ci_dict(boot, "tv_p1"),
            "tv_p2": _ci_dict(boot, "tv_p2"),
            "tv_avg_mean": float(tv_avg.mean()),
            "tv_avg_median": float(np.median(tv_avg)),
            "tv_avg_min": float(tv_avg.min()),
            "tv_avg_max": float(tv_avg.max()),
            "support_overlap_p1": _ci_dict(boot, "support_overlap_p1"),
            "support_overlap_p2": _ci_dict(boot, "support_overlap_p2"),
            "bc_mass_on_nash_p1": _ci_dict(boot, "bc_mass_on_nash_p1"),
            "bc_mass_on_nash_p2": _ci_dict(boot, "bc_mass_on_nash_p2"),
            "nash_supp_size_p1_mean": float(nash_supp_p1.mean()),
            "nash_supp_size_p2_mean": float(nash_supp_p2.mean()),
            "bc_supp_size_p1_mean": float(bc_supp_p1.mean()),
            "bc_supp_size_p2_mean": float(bc_supp_p2.mean()),
        },
        "regression": {
            "tv_vs_exploit": {
                "pearson_r": float(r_tv_exploit),
                "p_value": float(p_tv_exploit),
                "slope": float(slope),
                "intercept": float(intercept),
            },
            "supp_size_vs_exploit": {
                "pearson_r": float(r_supp_exploit),
                "p_value": float(p_supp_exploit),
            },
            "game_size_vs_exploit": {
                "pearson_r": float(r_size_exploit),
                "p_value": float(p_size_exploit),
            },
        },
        "bootstrap": boot,
    }


def _ci_dict(boot: dict, key: str) -> dict:
    """Extract mean + 95% CI from bootstrap results."""
    b = boot[key]
    return {
        "mean": b["mean"],
        "mean_ci": [b["mean_ci_lo"], b["mean_ci_hi"]],
        "median": b["median"],
        "median_ci": [b["median_ci_lo"], b["median_ci_hi"]],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Value decomposition + strategy distances")
    parser.add_argument("--bc_ckpt", required=True)
    parser.add_argument("--dyn_ckpt", required=True)
    parser.add_argument("--test_split", required=True)
    parser.add_argument("--vocab_path", required=True)
    parser.add_argument("--n_matchups", type=int, default=500)
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

    # Phase 0: GPU — build payoff matrices + BC strategies
    rng = np.random.RandomState(args.seed)
    indices = rng.choice(len(dataset), size=min(args.n_matchups, len(dataset)), replace=False)

    t0 = time.time()
    matchups = _build_matchups(bc_model, dyn_model, dataset, indices, device)
    print(f"Phase 0 time: {time.time() - t0:.1f}s")

    # Phase 1: CPU parallel — Nash + analysis
    print(f"\nPhase 1: Nash + cross-play + distances ({len(matchups)} matchups)...")
    t1 = time.time()

    matchup_args = [(m["idx"], m["R"], m["bc_p1"], m["bc_p2"]) for m in matchups]
    details = []

    if args.n_workers > 1 and len(matchup_args) > 1:
        with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
            futures = {
                executor.submit(_analyze_single_matchup, ma): ma[0]
                for ma in matchup_args
            }
            for future in tqdm(
                as_completed(futures), total=len(futures),
                desc="Phase 1: Nash + analysis (CPU parallel)",
            ):
                result = future.result()
                if result is not None:
                    details.append(result)
    else:
        for ma in tqdm(matchup_args, desc="Phase 1: Nash + analysis"):
            result = _analyze_single_matchup(ma)
            if result is not None:
                details.append(result)

    print(f"Phase 1 time: {time.time() - t1:.1f}s")
    print(f"  {len(details)} matchups solved (of {len(matchups)} built)")

    if len(details) == 0:
        print("ERROR: No matchups solved!")
        return

    # Phase 2: Aggregate
    print("\nPhase 2: Aggregation + regression...")
    results = _aggregate(details)

    # Print summary
    vd = results["value_decomposition"]
    cpg = results["cross_play_gaps"]
    sd = results["strategy_distances"]
    reg = results["regression"]

    print(f"\n{'='*60}")
    print(f"VALUE DECOMPOSITION ({results['n_matchups']} matchups)")
    print(f"{'='*60}")
    print(f"  V* (Nash value):         {vd['game_value']['mean']:.4f}  "
          f"CI {vd['game_value']['mean_ci']}")
    print(f"  BC-vs-BC:                {vd['bc_vs_bc']['mean']:.4f}  "
          f"CI {vd['bc_vs_bc']['mean_ci']}")
    print(f"  BC_P1 vs Nash_P2:        {vd['bc_p1_vs_nash_p2']['mean']:.4f}  "
          f"CI {vd['bc_p1_vs_nash_p2']['mean_ci']}")
    print(f"  Nash_P1 vs BC_P2:        {vd['nash_p1_vs_bc_p2']['mean']:.4f}  "
          f"CI {vd['nash_p1_vs_bc_p2']['mean_ci']}")
    print(f"  BC worst-case:           {vd['bc_worst_case']['mean']:.4f}")
    print(f"  Best-response to BC:     {vd['best_response_to_bc']['mean']:.4f}")
    print(f"  BC-vs-BC minus Nash:     {vd['bc_vs_bc_minus_nash']['mean']:.4f}  "
          f"CI {vd['bc_vs_bc_minus_nash']['mean_ci']}")

    print(f"\n  Cross-play gaps:")
    print(f"    P1 gap (BC_P1 vs Nash_P2 - V*): {cpg['cross_gap_p1']['mean']:.4f}  "
          f"CI {cpg['cross_gap_p1']['mean_ci']}")
    print(f"    P2 gap (Nash_P1 vs BC_P2 - V*): {cpg['cross_gap_p2']['mean']:.4f}  "
          f"CI {cpg['cross_gap_p2']['mean_ci']}")
    print(f"    |P1 gap| mean: {cpg['abs_cross_gap_p1_mean']:.4f}")
    print(f"    |P2 gap| mean: {cpg['abs_cross_gap_p2_mean']:.4f}")
    print(f"    Same sign: {cpg['n_same_sign']}, Opposite: {cpg['n_opposite_sign']}, "
          f"Zero: {cpg['n_zero']}")

    if cpg["cross_gap_p1"]["mean"] < -0.1 and cpg["cross_gap_p2"]["mean"] > 0.1:
        print(f"  --> ERROR CANCELLATION: P1 loses, P2 gives away, effects cancel")
    elif abs(cpg["cross_gap_p1"]["mean"]) < 0.15 and abs(cpg["cross_gap_p2"]["mean"]) < 0.15:
        print(f"  --> INDEPENDENT NEAR-OPTIMALITY: each player ≈ Nash value in cross-play")
    else:
        print(f"  --> MIXED: partial cancellation + partial near-optimality")

    print(f"\n{'='*60}")
    print(f"STRATEGY DISTANCES")
    print(f"{'='*60}")
    print(f"  TV distance P1: {sd['tv_p1']['mean']:.4f}  CI {sd['tv_p1']['mean_ci']}")
    print(f"  TV distance P2: {sd['tv_p2']['mean']:.4f}  CI {sd['tv_p2']['mean_ci']}")
    print(f"  TV avg: mean={sd['tv_avg_mean']:.4f}, median={sd['tv_avg_median']:.4f}, "
          f"range=[{sd['tv_avg_min']:.4f}, {sd['tv_avg_max']:.4f}]")
    print(f"  Support overlap P1: {sd['support_overlap_p1']['mean']:.4f}")
    print(f"  Support overlap P2: {sd['support_overlap_p2']['mean']:.4f}")
    print(f"  BC mass on Nash support P1: {sd['bc_mass_on_nash_p1']['mean']:.4f}")
    print(f"  BC mass on Nash support P2: {sd['bc_mass_on_nash_p2']['mean']:.4f}")
    print(f"  Nash support size: P1={sd['nash_supp_size_p1_mean']:.1f}, "
          f"P2={sd['nash_supp_size_p2_mean']:.1f}")
    print(f"  BC support size: P1={sd['bc_supp_size_p1_mean']:.1f}, "
          f"P2={sd['bc_supp_size_p2_mean']:.1f}")

    print(f"\n  Regression:")
    print(f"    TV vs exploit: r={reg['tv_vs_exploit']['pearson_r']:.4f} "
          f"(p={reg['tv_vs_exploit']['p_value']:.2e}), "
          f"slope={reg['tv_vs_exploit']['slope']:.4f}")
    print(f"    Support size vs exploit: r={reg['supp_size_vs_exploit']['pearson_r']:.4f} "
          f"(p={reg['supp_size_vs_exploit']['p_value']:.2e})")
    print(f"    Game size vs exploit: r={reg['game_size_vs_exploit']['pearson_r']:.4f} "
          f"(p={reg['game_size_vs_exploit']['p_value']:.2e})")

    # Save
    # Strip non-serializable bootstrap from main output (it's in the full JSON)
    with open(out_dir / "decomposition.json", "w") as f:
        json.dump(results, f, indent=2)

    # Per-matchup details for scatter plots
    with open(out_dir / "matchup_details.jsonl", "w") as f:
        for d in details:
            f.write(json.dumps(d) + "\n")

    print(f"\nTotal time: {time.time() - t0:.1f}s")
    print(f"Results saved to {out_dir}/")


if __name__ == "__main__":
    main()
