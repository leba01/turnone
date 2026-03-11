"""Qualitative analysis of CQL vs BC policy differences.

Extracts WHERE the 60% distribution shift lives:
  - Tera usage rates (CQL vs BC)
  - Offensive vs defensive targeting patterns
  - Joint action patterns (both-offensive, mixed, both-support)
  - SVD projection: is the shift payoff-relevant or null-space?

Requires the same checkpoints as cql_eval.py. Runs on CPU (inference only).

Usage:
    python scripts/cql_qualitative.py \
        --cql_ckpt runs/cql_001/best.pt \
        --bc_ckpt runs/bc_001/best.pt \
        --dyn_ckpt runs/dyn_001/best.pt \
        --test_split data/assembled/test.jsonl \
        --vocab_path runs/bc_001/vocab.json \
        --n_matchups 500 \
        --out_dir results/cql_qualitative/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from turnone.data.dataset import Turn1Dataset, Vocab
from turnone.data.action_space import slot_to_move_target, TERA_NONE
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
    """Convert factored Q-values to a probability vector over joint actions."""
    if not valid_actions:
        return np.zeros(0, dtype=np.float64)
    q_joint = np.zeros(len(valid_actions), dtype=np.float64)
    for i, (sa, sb, tf) in enumerate(valid_actions):
        q_joint[i] = q_a[sa] + q_b[sb] + q_tera[tf]
    q_joint = q_joint / tau
    q_max = np.max(q_joint)
    if not np.isfinite(q_max):
        q_max = 0.0
    e = np.exp(q_joint - q_max)
    return e / e.sum()


def classify_actions(
    strategy: np.ndarray,
    valid_actions: list[tuple[int, int, int]],
) -> dict[str, float]:
    """Classify a strategy's probability mass by behavioral categories.

    Returns dict with:
        tera_rate: P(tera_flag > 0)
        offensive_a: P(mon_a targets opponent)
        offensive_b: P(mon_b targets opponent)
        offensive_rate: P(at least one mon targets opponent)
        both_offensive: P(both mons target opponents)
        both_support: P(both mons target ally/self)
        mixed: P(one offensive, one support)
        target_opp_rate: avg fraction of mass on opponent-targeting actions
    """
    tera_mass = 0.0
    off_a_mass = 0.0
    off_b_mass = 0.0
    both_off_mass = 0.0
    both_sup_mass = 0.0
    mixed_mass = 0.0

    for i, (sa, sb, tf) in enumerate(valid_actions):
        p = strategy[i]
        if p < 1e-12:
            continue

        # Tera
        if tf != TERA_NONE:
            tera_mass += p

        # Targeting: target in {0, 1} = opponent, {2, 3} = ally/self
        _, tgt_a = slot_to_move_target(sa)
        _, tgt_b = slot_to_move_target(sb)
        a_off = tgt_a in (0, 1)
        b_off = tgt_b in (0, 1)

        if a_off:
            off_a_mass += p
        if b_off:
            off_b_mass += p

        if a_off and b_off:
            both_off_mass += p
        elif not a_off and not b_off:
            both_sup_mass += p
        else:
            mixed_mass += p

    return {
        "tera_rate": float(tera_mass),
        "offensive_a": float(off_a_mass),
        "offensive_b": float(off_b_mass),
        "target_opp_rate": float((off_a_mass + off_b_mass) / 2),
        "both_offensive": float(both_off_mass),
        "both_support": float(both_sup_mass),
        "mixed": float(mixed_mass),
    }


def svd_projection(
    delta: np.ndarray,
    R: np.ndarray,
    k: int = 3,
) -> dict[str, float]:
    """Project a strategy deviation onto R's SVD subspaces.

    Returns fraction of ||delta||^2 in top-k dimensions (payoff-relevant)
    and in null space (payoff-irrelevant).
    """
    U, S, Vt = np.linalg.svd(R, full_matrices=True)
    delta_norm_sq = float(np.dot(delta, delta))
    if delta_norm_sq < 1e-15:
        return {"in_top_k": 0.0, "in_null": 1.0, "delta_norm": 0.0}

    # Project onto top-k left singular vectors (column space of R)
    proj = U[:, :k].T @ delta  # (k,)
    proj_norm_sq = float(np.dot(proj, proj))
    frac_top_k = proj_norm_sq / delta_norm_sq

    return {
        "in_top_k": frac_top_k,
        "in_null": 1.0 - frac_top_k,
        "delta_norm": float(np.sqrt(delta_norm_sq)),
    }


def tv_distance(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.sum(np.abs(p - q)))


def main():
    parser = argparse.ArgumentParser(
        description="Qualitative analysis of CQL vs BC policy shift")
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

    # Sample matchups (same seed as cql_eval.py for consistency)
    rng = np.random.RandomState(args.seed)
    n = min(args.n_matchups, len(dataset))
    indices = rng.choice(len(dataset), size=n, replace=False)
    indices.sort()

    print(f"Evaluating {n} matchups...")
    per_matchup = []
    skipped = 0

    for idx in tqdm(indices, desc="Qualitative analysis"):
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

        # Build payoff matrix (needed for SVD projection)
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

        # BC strategy
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

        # CQL strategy
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

        # Classify BC and CQL behavior
        bc_class = classify_actions(bc_p1, actions_p1)
        cql_class = classify_actions(cql_p1, actions_p1)
        nash_class = classify_actions(nash_p1, actions_p1)

        # Exploitability
        bc_exploit = exploitability_from_nash(bc_p1, R, game_value, player=1)
        cql_exploit = exploitability_from_nash(cql_p1, R, game_value, player=1)

        # TV distances
        tv_cql_bc = tv_distance(cql_p1, bc_p1)
        tv_bc_nash = tv_distance(bc_p1, nash_p1)
        tv_cql_nash = tv_distance(cql_p1, nash_p1)

        # SVD projection of CQL-BC deviation
        delta_cql_bc = cql_p1 - bc_p1
        svd_cql_bc = svd_projection(delta_cql_bc, R, k=3)

        # SVD projection of BC-Nash deviation (for comparison)
        delta_bc_nash = bc_p1 - nash_p1
        svd_bc_nash = svd_projection(delta_bc_nash, R, k=3)

        per_matchup.append({
            "idx": int(idx),
            "n_actions_p1": len(actions_p1),
            "bc_exploit": bc_exploit,
            "cql_exploit": cql_exploit,
            "tv_cql_bc": tv_cql_bc,
            "tv_bc_nash": tv_bc_nash,
            "tv_cql_nash": tv_cql_nash,
            "bc": bc_class,
            "cql": cql_class,
            "nash": nash_class,
            "svd_cql_bc_shift": svd_cql_bc,
            "svd_bc_nash_shift": svd_bc_nash,
        })

    if skipped:
        print(f"Skipped {skipped} matchups (empty actions or LP failure)")

    if not per_matchup:
        print("No valid results!")
        return

    # ── Aggregate ──
    n_valid = len(per_matchup)

    def agg_stat(values):
        arr = np.array(values)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "median": float(np.median(arr)),
            "ci_lo": float(np.percentile(arr, 2.5)),
            "ci_hi": float(np.percentile(arr, 97.5)),
        }

    # Behavioral comparison
    behavioral = {}
    for metric in ["tera_rate", "target_opp_rate", "both_offensive",
                    "both_support", "mixed", "offensive_a", "offensive_b"]:
        bc_vals = [r["bc"][metric] for r in per_matchup]
        cql_vals = [r["cql"][metric] for r in per_matchup]
        nash_vals = [r["nash"][metric] for r in per_matchup]
        behavioral[metric] = {
            "bc": agg_stat(bc_vals),
            "cql": agg_stat(cql_vals),
            "nash": agg_stat(nash_vals),
        }

    # SVD projection of the CQL-BC shift
    svd_agg = {
        "cql_bc_in_top3": agg_stat(
            [r["svd_cql_bc_shift"]["in_top_k"] for r in per_matchup]),
        "cql_bc_in_null": agg_stat(
            [r["svd_cql_bc_shift"]["in_null"] for r in per_matchup]),
        "bc_nash_in_top3": agg_stat(
            [r["svd_bc_nash_shift"]["in_top_k"] for r in per_matchup]),
        "bc_nash_in_null": agg_stat(
            [r["svd_bc_nash_shift"]["in_null"] for r in per_matchup]),
    }

    # Exploitability
    exploit_agg = {
        "bc": agg_stat([r["bc_exploit"] for r in per_matchup]),
        "cql": agg_stat([r["cql_exploit"] for r in per_matchup]),
        "improvement_pct": agg_stat(
            [100 * (r["bc_exploit"] - r["cql_exploit"]) / max(r["bc_exploit"], 1e-8)
             for r in per_matchup]),
        "cql_better_count": sum(
            1 for r in per_matchup if r["cql_exploit"] < r["bc_exploit"]),
    }

    # TV distances
    tv_agg = {
        "cql_bc": agg_stat([r["tv_cql_bc"] for r in per_matchup]),
        "bc_nash": agg_stat([r["tv_bc_nash"] for r in per_matchup]),
        "cql_nash": agg_stat([r["tv_cql_nash"] for r in per_matchup]),
    }

    output = {
        "n_matchups": n_valid,
        "n_skipped": skipped,
        "temperature": args.temperature,
        "behavioral": behavioral,
        "svd_projection": svd_agg,
        "exploitability": exploit_agg,
        "tv_distances": tv_agg,
        "per_matchup": per_matchup,
    }

    out_path = out_dir / "cql_qualitative.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # ── Print summary ──
    print("\n" + "=" * 70)
    print("QUALITATIVE COMPARISON: BC vs CQL vs Nash")
    print("=" * 70)

    print(f"\n{'Metric':<25} {'BC':>10} {'CQL':>10} {'Nash':>10}")
    print("-" * 58)
    for metric in ["tera_rate", "target_opp_rate", "both_offensive",
                    "both_support", "mixed"]:
        bc_m = behavioral[metric]["bc"]["mean"]
        cql_m = behavioral[metric]["cql"]["mean"]
        nash_m = behavioral[metric]["nash"]["mean"]
        print(f"{metric:<25} {bc_m:10.3f} {cql_m:10.3f} {nash_m:10.3f}")

    print(f"\n{'SVD Projection':<35} {'Mean':>10}")
    print("-" * 48)
    print(f"{'CQL-BC shift in top-3 (payoff-rel)':<35} "
          f"{svd_agg['cql_bc_in_top3']['mean']:10.3f}")
    print(f"{'CQL-BC shift in null (payoff-irrel)':<35} "
          f"{svd_agg['cql_bc_in_null']['mean']:10.3f}")
    print(f"{'BC-Nash shift in top-3 (reference)':<35} "
          f"{svd_agg['bc_nash_in_top3']['mean']:10.3f}")
    print(f"{'BC-Nash shift in null (reference)':<35} "
          f"{svd_agg['bc_nash_in_null']['mean']:10.3f}")

    print(f"\nExploitability: BC {exploit_agg['bc']['mean']:.3f} → "
          f"CQL {exploit_agg['cql']['mean']:.3f} "
          f"({exploit_agg['improvement_pct']['mean']:.1f}% improvement)")
    print(f"CQL better in {exploit_agg['cql_better_count']}/{n_valid} matchups")
    print(f"TV(CQL, BC) = {tv_agg['cql_bc']['mean']:.3f}")


if __name__ == "__main__":
    main()
