"""Methodology audit for TurnOne findings.

Runs concrete checks to stress-test whether findings are real
or methodological artifacts before the milestone writeup.

Checks:
    1  (P0) BC-vs-BC ≈ Nash: is this trivially true because V* ≈ 0?
    2b (P0) How much BC probability mass stripped by strategic mask?
    6  (P0) Do LP solutions achieve claimed game value?
    11 (P0) Does Nash put weight on nonsensical ally-targeting actions?
    3  (P1) Correlated noise vs independent noise
    7  (P1) Field component magnitude + w_field=0 ablation
    8  (P2) Exploitability tail analysis
    10 (P2) Bootstrap independence (team reuse)

Usage:
    PYTHONPATH=. python scripts/audit.py \
        --bc_ckpt runs/bc_001/best.pt \
        --dyn_ckpt runs/dyn_001/best.pt \
        --test_split data/assembled/test.jsonl \
        --vocab_path runs/bc_001/vocab.json \
        --matchup_details results/matchup_details.jsonl \
        --out_dir results/audit/

    Fast-only (CPU checks 1, 8, 10 — ~30 sec):
    PYTHONPATH=. python scripts/audit.py --fast_only \
        --matchup_details results/matchup_details.jsonl \
        --test_split data/assembled/test.jsonl \
        --vocab_path runs/bc_001/vocab.json \
        --bc_ckpt x --dyn_ckpt x --out_dir results/audit/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats
from tqdm import tqdm


# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────

def load_matchup_details(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


# ──────────────────────────────────────────────────────────
# Check 1 — BC-vs-BC ≈ Nash tracking  (P0)
# ──────────────────────────────────────────────────────────

def check1_bc_vs_bc_tracking(matchups: list[dict]) -> dict:
    """Is 'BC-vs-BC ≈ Nash' trivially true because V* ≈ 0?

    Regression BC-vs-BC = alpha + beta * V*.
    If beta ≈ 1 and the gap is small even for extreme matchups,
    the tracking is genuine — not just '0 ≈ 0'.
    """
    gv = np.array([m["game_value"] for m in matchups])
    bcbc = np.array([m["bc_vs_bc"] for m in matchups])
    gap = bcbc - gv
    abs_gv = np.abs(gv)

    # Quintile analysis by |V*|
    edges = np.percentile(abs_gv, np.linspace(0, 100, 6))
    quintiles = []
    for i in range(5):
        lo, hi = edges[i], edges[i + 1]
        mask = (abs_gv >= lo) & (abs_gv <= hi) if i == 4 else \
               (abs_gv >= lo) & (abs_gv < hi)
        if mask.sum() == 0:
            continue
        q_gv, q_bcbc, q_gap = gv[mask], bcbc[mask], gap[mask]
        corr = float(np.corrcoef(q_gv, q_bcbc)[0, 1]) if len(q_gv) > 2 else float("nan")
        quintiles.append({
            "quintile": i + 1,
            "abs_gv_range": [float(lo), float(hi)],
            "n": int(mask.sum()),
            "gap_mean": float(q_gap.mean()),
            "gap_abs_mean": float(np.abs(q_gap).mean()),
            "correlation": corr,
        })

    # Linear regression: BC-vs-BC = alpha + beta * V*
    slope, intercept, r_value, p_value, _ = stats.linregress(gv, bcbc)

    # Top-50 by |V*|
    top50_idx = np.argsort(abs_gv)[-50:]
    top50_gap = gap[top50_idx]

    # Top quintile gap
    top_q_gap = quintiles[-1]["gap_abs_mean"] if quintiles else float("nan")

    beta_pass = 0.7 <= slope <= 1.3
    top_gap_pass = top_q_gap < 0.3
    verdict = "PASS" if (beta_pass and top_gap_pass) else "FAIL"

    return {
        "check": "1_bc_vs_bc_tracking", "verdict": verdict,
        "regression": {
            "beta": float(slope), "alpha": float(intercept),
            "r_squared": float(r_value ** 2), "p_value": float(p_value),
        },
        "quintiles": quintiles,
        "top50_by_abs_gv": {
            "gap_mean": float(top50_gap.mean()),
            "gap_abs_mean": float(np.abs(top50_gap).mean()),
        },
        "overall": {
            "gap_mean": float(gap.mean()), "gap_std": float(gap.std()),
            "abs_gv_median": float(np.median(abs_gv)),
        },
        "criteria": {
            "beta_in_0.7_1.3": beta_pass,
            "top_quintile_gap_lt_0.3": top_gap_pass,
        },
    }


# ──────────────────────────────────────────────────────────
# Check 2b — Stripped BC probability mass  (P0)
# ──────────────────────────────────────────────────────────

def check2b_stripped_mass(bc_model, dataset, matchup_indices, device) -> dict:
    """How much BC probability mass is stripped by the strategic mask?

    BC trains with target=3 allowed for single-target moves;
    Nash enumeration excludes target=3 via strategic_mask.
    bc_strategy_from_logits renormalizes silently.
    """
    import torch
    from turnone.game.payoff import enumerate_joint_actions

    stripped_a_list, stripped_b_list, joint_total_list = [], [], []

    for idx in tqdm(matchup_indices, desc="Check 2b: stripped mass"):
        example = dataset[idx]
        mask_a = example["mask_a"].numpy()
        mask_b = example["mask_b"].numpy()
        strat_a = example["strategic_mask_a"].numpy()
        strat_b = example["strategic_mask_b"].numpy()

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

        prob_a = _softmax(bc_out["logits_a"][0].cpu().numpy())
        prob_b = _softmax(bc_out["logits_b"][0].cpu().numpy())
        prob_tera = _softmax(bc_out["logits_tera"][0].cpu().numpy())

        # Slots in BC mask but not strategic mask
        diff_a = mask_a & ~strat_a
        diff_b = mask_b & ~strat_b
        stripped_a_list.append(float(prob_a[diff_a].sum()))
        stripped_b_list.append(float(prob_b[diff_b].sum()))

        # Joint total (pre-normalization sum in bc_strategy_from_logits)
        valid_actions = enumerate_joint_actions(strat_a, strat_b, include_tera=True)
        joint_total = sum(
            prob_a[sa] * prob_b[sb] * prob_tera[tf]
            for sa, sb, tf in valid_actions
        )
        joint_total_list.append(float(joint_total))

    sa = np.array(stripped_a_list)
    sb = np.array(stripped_b_list)
    jt = np.array(joint_total_list)

    med_stripped = float(np.median(np.maximum(sa, sb)))
    med_total = float(np.median(jt))
    mass_pass = med_stripped < 0.05
    total_pass = med_total > 0.90
    verdict = "PASS" if (mass_pass and total_pass) else "FAIL"

    return {
        "check": "2b_stripped_mass", "verdict": verdict,
        "stripped_a": {
            "mean": float(sa.mean()), "median": float(np.median(sa)),
            "p95": float(np.percentile(sa, 95)), "max": float(sa.max()),
        },
        "stripped_b": {
            "mean": float(sb.mean()), "median": float(np.median(sb)),
            "p95": float(np.percentile(sb, 95)), "max": float(sb.max()),
        },
        "joint_total": {
            "mean": float(jt.mean()), "median": med_total,
            "p5": float(np.percentile(jt, 5)), "min": float(jt.min()),
        },
        "criteria": {
            "median_per_mon_stripped_lt_5pct": mass_pass,
            "median_joint_total_gt_90pct": total_pass,
        },
    }


# ──────────────────────────────────────────────────────────
# Check 3 — Correlated noise vs independent noise  (P1)
# ──────────────────────────────────────────────────────────

def check3_correlated_noise(payoff_data: list[dict],
                            reward_mae: float,
                            existing_noise: dict) -> dict:
    """Correlated noise (row+col+entry) vs i.i.d. Gaussian per cell.

    Real dynamics errors are correlated (same encoder, shared state).
    If correlated noise produces much worse Nash instability than i.i.d.,
    the current noise floor is misleadingly optimistic.
    """
    from turnone.game.nash import solve_nash_lp
    from turnone.game.exploitability import exploitability_from_nash

    # Part 1: bias-shift sanity check
    # Adding a constant to R shifts game value but cannot change strategies.
    bias = 0.26  # measured reward bias from dynamics_metrics.json
    bias_tvs = []
    for item in payoff_data:
        R = item["R"]
        try:
            clean_p1, _, _ = solve_nash_lp(R)
            shift_p1, _, _ = solve_nash_lp(R + bias)
        except ValueError:
            continue
        bias_tvs.append(float(0.5 * np.abs(clean_p1 - shift_p1).sum()))

    bias_tv_arr = np.array(bias_tvs) if bias_tvs else np.array([])
    bias_pass = float(np.median(bias_tv_arr)) < 1e-3 if len(bias_tv_arr) > 0 else False

    # Part 2: correlated vs i.i.d. noise at 1.0× MAE
    noise_std = reward_mae
    n_trials = 20
    corr_tv, corr_ex, iid_tv, iid_ex = [], [], [], []

    for item in tqdm(payoff_data, desc="Check 3: noise experiment"):
        R, bc_strat = item["R"], item["bc_p1"]
        n1, n2 = R.shape
        rng = np.random.RandomState(42 + item["idx"])

        try:
            clean_p1, _, clean_value = solve_nash_lp(R)
        except ValueError:
            continue

        for _ in range(n_trials):
            # Correlated: row + col + entry, total variance = noise_std^2
            # row_var=0.25σ², col_var=0.25σ², entry_var=0.5σ²
            row_n = rng.randn(n1, 1).astype(np.float32) * (0.5 * noise_std)
            col_n = rng.randn(1, n2).astype(np.float32) * (0.5 * noise_std)
            ent_n = rng.randn(n1, n2).astype(np.float32) * (np.sqrt(0.5) * noise_std)
            R_corr = R + row_n + col_n + ent_n

            # I.i.d. (same total variance)
            R_iid = R + rng.randn(n1, n2).astype(np.float32) * noise_std

            for R_noisy, tv_list, ex_list in [
                (R_corr, corr_tv, corr_ex),
                (R_iid, iid_tv, iid_ex),
            ]:
                try:
                    noisy_p1, _, _ = solve_nash_lp(R_noisy)
                    tv_list.append(float(0.5 * np.abs(clean_p1 - noisy_p1).sum()))
                    ex_list.append(float(exploitability_from_nash(
                        noisy_p1, R, clean_value, player=1)))
                except ValueError:
                    pass

    def _stats(arr):
        if not arr:
            return {"n": 0, "mean": None, "std": None}
        a = np.array(arr)
        return {"n": len(a), "mean": float(a.mean()), "std": float(a.std())}

    corr_tv_s, iid_tv_s = _stats(corr_tv), _stats(iid_tv)
    corr_ex_s, iid_ex_s = _stats(corr_ex), _stats(iid_ex)

    tv_ratio = (corr_tv_s["mean"] / max(iid_tv_s["mean"], 1e-10)
                if corr_tv_s["mean"] is not None and iid_tv_s["mean"] is not None
                else float("nan"))
    ex_ratio = (corr_ex_s["mean"] / max(iid_ex_s["mean"], 1e-10)
                if corr_ex_s["mean"] is not None and iid_ex_s["mean"] is not None
                else float("nan"))

    ratio_pass = (np.isfinite(tv_ratio) and tv_ratio < 2.0
                  and np.isfinite(ex_ratio) and ex_ratio < 2.0)
    verdict = "PASS" if (bias_pass and ratio_pass) else "FAIL"

    return {
        "check": "3_correlated_noise", "verdict": verdict,
        "bias_shift": {
            "bias": bias,
            "tv_mean": float(bias_tv_arr.mean()) if len(bias_tv_arr) > 0 else None,
            "tv_median": float(np.median(bias_tv_arr)) if len(bias_tv_arr) > 0 else None,
            "pass": bias_pass,
        },
        "correlated_1x": {"tv": corr_tv_s, "exploit": corr_ex_s},
        "iid_1x": {"tv": iid_tv_s, "exploit": iid_ex_s},
        "ratios": {
            "tv_corr_over_iid": float(tv_ratio),
            "exploit_corr_over_iid": float(ex_ratio),
        },
        "existing_iid_1x": existing_noise.get("scale_1.0", {}),
        "criteria": {
            "bias_tv_lt_1e3": bias_pass,
            "corr_within_2x_iid": ratio_pass,
        },
    }


# ──────────────────────────────────────────────────────────
# Check 6 — LP solution verification  (P0)
# ──────────────────────────────────────────────────────────

def check6_lp_verification(payoff_data: list[dict],
                           stored_by_idx: dict[int, dict]) -> dict:
    """Verify that Nash LP solutions actually achieve the claimed game value.

    For each rebuilt payoff matrix:
    - min_j(nash_p1 @ R)_j should ≈ game_value   (P1's guarantee)
    - max_i(R @ nash_p2)_i should ≈ game_value   (P2's guarantee)
    Also compare rebuilt game value to the stored value in matchup_details.
    """
    from turnone.game.nash import solve_nash_lp

    results = []
    for item in tqdm(payoff_data, desc="Check 6: LP verification"):
        R, idx = item["R"], item["idx"]
        try:
            nash_p1, nash_p2, game_value = solve_nash_lp(R)
        except ValueError:
            continue

        p1_min = float(np.min(nash_p1 @ R))
        p2_max = float(np.max(R @ nash_p2))
        err_p1 = abs(p1_min - game_value)
        err_p2 = abs(p2_max - game_value)

        stored_gv = stored_by_idx.get(idx, {}).get("game_value")
        rebuild_err = abs(game_value - stored_gv) if stored_gv is not None else None

        results.append({
            "idx": idx,
            "game_value": float(game_value),
            "p1_min_payoff": float(p1_min),
            "p2_max_payoff": float(p2_max),
            "verification_error_p1": float(err_p1),
            "verification_error_p2": float(err_p2),
            "max_error": float(max(err_p1, err_p2)),
            "rebuild_vs_stored": float(rebuild_err) if rebuild_err is not None else None,
        })

    if not results:
        return {"check": "6_lp_verification", "verdict": "FAIL", "error": "no matchups solved"}

    max_errors = np.array([r["max_error"] for r in results])
    pct_1e3 = float((max_errors < 1e-3).mean())
    pct_1e2 = float((max_errors < 1e-2).mean())
    pass_strict = pct_1e3 > 0.99

    rebuild_errs = [r["rebuild_vs_stored"] for r in results if r["rebuild_vs_stored"] is not None]
    rebuild_arr = np.array(rebuild_errs) if rebuild_errs else np.array([])

    verdict = "PASS" if pass_strict else ("BORDERLINE" if pct_1e3 > 0.95 else "FAIL")

    return {
        "check": "6_lp_verification", "verdict": verdict,
        "n_matchups": len(results),
        "max_error_mean": float(max_errors.mean()),
        "max_error_max": float(max_errors.max()),
        "max_error_p95": float(np.percentile(max_errors, 95)),
        "pct_within_1e-3": pct_1e3,
        "pct_within_1e-2": pct_1e2,
        "rebuild_vs_stored": {
            "n": len(rebuild_arr),
            "max": float(rebuild_arr.max()) if len(rebuild_arr) > 0 else None,
            "mean": float(rebuild_arr.mean()) if len(rebuild_arr) > 0 else None,
        },
        "worst_5": sorted(results, key=lambda r: -r["max_error"])[:5],
        "criteria": {"gt_99pct_within_1e3": pass_strict},
    }


# ──────────────────────────────────────────────────────────
# Check 7 — Field component + w_field=0 ablation  (P1)
# ──────────────────────────────────────────────────────────

def check7_field_ablation(dyn_model, bc_model, dataset, matchup_indices,
                          device, stored_by_idx) -> dict:
    """Field component magnitude and w_field=0 ablation.

    Part 1: Decompose ground-truth reward into HP/KO/field on 2000 samples.
    Part 2: Rebuild 200 payoff matrices with w_field=0, compare exploitability.
    """
    import torch
    from turnone.rl.reward import compute_reward
    from turnone.game.payoff import enumerate_joint_actions, build_payoff_matrix
    from turnone.game.exploitability import (
        bc_strategy_from_logits, exploitability_from_nash, compute_strategy_values,
    )
    from turnone.game.nash import solve_nash_lp

    # ---- Part 1: reward decomposition ----
    n_sample = min(2000, len(dataset))
    rng = np.random.RandomState(123)
    sample_idx = rng.choice(len(dataset), size=n_sample, replace=False)

    hp_d, ko_f, f_bef, f_aft = [], [], [], []
    for idx in sample_idx:
        ex = dataset[idx]
        hp_d.append(ex["hp_delta"].numpy())
        ko_f.append(ex["ko_flags"].numpy())
        f_bef.append(ex["field_state"].numpy())
        f_aft.append(ex["field_after"].numpy())

    hp_delta = np.stack(hp_d)
    ko_flags = np.stack(ko_f)
    field_before = np.stack(f_bef)
    field_after = np.stack(f_aft)

    r_hp = compute_reward(hp_delta, ko_flags, field_before, field_after, 1.0, 0.0, 0.0)
    r_ko = compute_reward(hp_delta, ko_flags, field_before, field_after, 0.0, 3.0, 0.0)
    r_field = compute_reward(hp_delta, ko_flags, field_before, field_after, 0.0, 0.0, 0.5)

    abs_hp, abs_ko, abs_field = np.abs(r_hp), np.abs(r_ko), np.abs(r_field)
    total = abs_hp + abs_ko + abs_field
    field_frac = abs_field / np.where(total > 0, total, 1.0)

    # ---- Part 2: w_field=0 ablation on 200 matchups ----
    abl_indices = matchup_indices[:200]
    ablated_results = []

    for idx in tqdm(abl_indices, desc="Check 7: w_field=0 matrices"):
        example = dataset[idx]
        strat_a = example["strategic_mask_a"].numpy()
        strat_b = example["strategic_mask_b"].numpy()
        opp_strat_a = example["opp_strategic_mask_a"].numpy()
        opp_strat_b = example["opp_strategic_mask_b"].numpy()

        actions_p1 = enumerate_joint_actions(strat_a, strat_b, include_tera=True)
        actions_p2 = enumerate_joint_actions(opp_strat_a, opp_strat_b, include_tera=True)
        if not actions_p1 or not actions_p2:
            continue

        state = {k: example[k] for k in ("team_a", "team_b", "lead_a", "lead_b", "field_state")}

        R_abl = build_payoff_matrix(
            dyn_model, state, actions_p1, actions_p2,
            example["field_state"].numpy(), device,
            reward_weights={"w_hp": 1.0, "w_ko": 3.0, "w_field": 0.0},
        )

        # P1 BC strategy
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
            bc_out["logits_tera"][0].cpu().numpy(), actions_p1,
        )

        # P2 BC strategy (perspective swap)
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
            bc_out_p2["logits_tera"][0].cpu().numpy(), actions_p2,
        )

        try:
            _, _, gv = solve_nash_lp(R_abl)
        except ValueError:
            continue

        exploit = exploitability_from_nash(bc_p1, R_abl, gv, player=1)
        tri = compute_strategy_values(bc_p1, bc_p2, R_abl, gv)

        stored = stored_by_idx.get(idx, {})
        ablated_results.append({
            "idx": idx,
            "ablated_exploit": float(exploit),
            "ablated_bcbc": float(tri["bc_vs_bc"]),
            "ablated_gv": float(gv),
            "baseline_exploit": stored.get("bc_exploitability"),
            "baseline_bcbc": stored.get("bc_vs_bc"),
        })

    # Compare matched pairs
    abl_ex = np.array([r["ablated_exploit"] for r in ablated_results
                       if r["baseline_exploit"] is not None])
    bas_ex = np.array([r["baseline_exploit"] for r in ablated_results
                       if r["baseline_exploit"] is not None])

    if len(bas_ex) > 0:
        exploit_change = abs(abl_ex.mean() - bas_ex.mean()) / max(abs(bas_ex.mean()), 1e-10)
    else:
        exploit_change = float("nan")

    ff_mean = float(np.mean(field_frac))
    frac_pass = ff_mean < 0.10
    change_pass = exploit_change < 0.15 if np.isfinite(exploit_change) else False
    frac_relaxed = ff_mean < 0.25
    change_relaxed = exploit_change < 0.30 if np.isfinite(exploit_change) else False

    verdict = ("PASS" if frac_pass and change_pass
               else "BORDERLINE" if frac_relaxed and change_relaxed
               else "FAIL")

    return {
        "check": "7_field_ablation", "verdict": verdict,
        "field_magnitude": {
            "hp_mean_abs": float(abs_hp.mean()),
            "ko_mean_abs": float(abs_ko.mean()),
            "field_mean_abs": float(abs_field.mean()),
            "field_fraction_mean": ff_mean,
            "field_fraction_median": float(np.median(field_frac)),
            "field_fraction_p95": float(np.percentile(field_frac, 95)),
            "n_samples": n_sample,
        },
        "ablation": {
            "n_matchups": len(ablated_results),
            "ablated_exploit_mean": float(abl_ex.mean()) if len(abl_ex) > 0 else None,
            "baseline_exploit_mean": float(bas_ex.mean()) if len(bas_ex) > 0 else None,
            "exploit_change_pct": float(exploit_change * 100) if np.isfinite(exploit_change) else None,
        },
        "criteria": {
            "field_fraction_lt_10pct": frac_pass,
            "exploit_change_lt_15pct": change_pass,
        },
    }


# ──────────────────────────────────────────────────────────
# Check 8 — Exploitability tail analysis  (P2)
# ──────────────────────────────────────────────────────────

def check8_tail_analysis(matchups: list[dict]) -> dict:
    """Are exploitability outliers inflating the mean?"""
    exploits = np.array([m["bc_exploitability"] for m in matchups])
    gv = np.array([m["game_value"] for m in matchups])
    na_p1 = np.array([m["n_actions_p1"] for m in matchups])
    sup_p1 = np.array([m["nash_support_p1"] for m in matchups])

    pcts = {f"p{p}": float(np.percentile(exploits, p))
            for p in [5, 10, 25, 50, 75, 90, 95]}

    n = len(exploits)
    trim_n = max(1, int(n * 0.05))
    trimmed = np.sort(exploits)[trim_n:-trim_n]
    trimmed_mean = float(trimmed.mean())
    full_mean = float(exploits.mean())
    trim_change = abs(trimmed_mean - full_mean) / max(abs(full_mean), 1e-10)

    tail = exploits > 2.0
    n_tail = int(tail.sum())
    tail_info = {"n_tail": n_tail, "pct_tail": float(n_tail / n)}
    if n_tail > 0:
        tail_info.update({
            "tail_mean_exploit": float(exploits[tail].mean()),
            "tail_mean_gv": float(gv[tail].mean()),
            "tail_mean_actions_p1": float(na_p1[tail].mean()),
            "tail_mean_support_p1": float(sup_p1[tail].mean()),
            "rest_mean_actions_p1": float(na_p1[~tail].mean()),
            "rest_mean_support_p1": float(sup_p1[~tail].mean()),
        })

    trim_pass = trim_change < 0.15
    verdict = "PASS" if trim_pass else "FAIL"

    return {
        "check": "8_tail_analysis", "verdict": verdict,
        "full_mean": full_mean, "trimmed_mean_5pct": trimmed_mean,
        "trim_change_pct": float(trim_change * 100),
        "percentiles": pcts, "tail_gt_2": tail_info,
        "criteria": {"trimmed_within_15pct": trim_pass},
    }


# ──────────────────────────────────────────────────────────
# Check 10 — Team reuse  (P2)
# ──────────────────────────────────────────────────────────

def check10_team_reuse(dataset, matchup_indices: list[int]) -> dict:
    """Are sampled matchups independent, or do teams recur?"""
    team_hashes = []
    for idx in matchup_indices:
        ex = dataset[idx]
        team_hashes.append(hashlib.md5(ex["team_a"].numpy().tobytes()).hexdigest())
        team_hashes.append(hashlib.md5(ex["team_b"].numpy().tobytes()).hexdigest())

    unique = len(set(team_hashes))
    total = len(team_hashes)
    reuse = 1.0 - unique / total

    counts = Counter(team_hashes)
    max_app = max(counts.values())
    teams_2plus = sum(1 for c in counts.values() if c >= 2)

    pass_ = reuse < 0.10
    verdict = "PASS" if pass_ else ("BORDERLINE" if reuse < 0.30 else "FAIL")

    return {
        "check": "10_team_reuse", "verdict": verdict,
        "total_team_slots": total, "unique_teams": unique,
        "reuse_rate": float(reuse),
        "unique_team_a": len(set(team_hashes[::2])),
        "unique_team_b": len(set(team_hashes[1::2])),
        "max_appearances": max_app, "teams_appearing_2plus": teams_2plus,
        "criteria": {"reuse_lt_10pct": pass_},
    }


# ──────────────────────────────────────────────────────────
# Check 11 — Nash action quality  (P0)
# ──────────────────────────────────────────────────────────

def check11_action_quality(payoff_data: list[dict], dataset) -> dict:
    """Does Nash put weight on nonsensical ally-targeting actions?

    Single-target moves (Fake Out, Thunderbolt, etc.) aimed at your own
    ally (target=2) are almost always dominated. If Nash assigns significant
    weight to these, something is wrong with the payoff matrix.

    Exceptions: some single-target moves legitimately target allies
    (Pollen Puff heals ally, Beat Up triggers Justified, etc.).
    """
    from turnone.game.nash import solve_nash_lp
    from turnone.data.action_space import get_target_category, DUAL_PURPOSE_MOVES

    ALLY_VALID_SINGLE = DUAL_PURPOSE_MOVES

    per_matchup = []

    for item in tqdm(payoff_data, desc="Check 11: action quality"):
        R = item["R"]
        idx = item["idx"]
        actions_p1 = item["actions_p1"]

        try:
            nash_p1, _, _ = solve_nash_lp(R)
        except ValueError:
            continue

        # Raw example to get move names
        raw_ex = dataset.examples[idx]
        lead_a_idx = raw_ex["lead_indices_a"][0]
        lead_b_idx = raw_ex["lead_indices_a"][1]
        moves_a = raw_ex["team_a"][lead_a_idx]["moves"]
        moves_b = raw_ex["team_a"][lead_b_idx]["moves"]

        ally_attack_weight = 0.0
        ally_attack_count = 0
        flagged = []

        for ai, (slot_a, slot_b, tera) in enumerate(actions_p1):
            target_a = slot_a % 4
            target_b = slot_b % 4
            move_name_a = moves_a[slot_a // 4]
            move_name_b = moves_b[slot_b // 4]
            cat_a = get_target_category(move_name_a)
            cat_b = get_target_category(move_name_b)

            # Suspicious: single-target offensive move aimed at ally
            # (exclude moves with legitimate ally-targeting use)
            sus_a = (target_a == 2 and cat_a == "single"
                     and move_name_a not in ALLY_VALID_SINGLE)
            sus_b = (target_b == 2 and cat_b == "single"
                     and move_name_b not in ALLY_VALID_SINGLE)

            if sus_a or sus_b:
                ally_attack_count += 1
                w = float(nash_p1[ai])
                ally_attack_weight += w
                if w > 0.01:
                    flagged.append({
                        "nash_weight": round(w, 4),
                        "slot_a": slot_a, "slot_b": slot_b, "tera": tera,
                        "sus_a": f"{move_name_a}->ally" if sus_a else None,
                        "sus_b": f"{move_name_b}->ally" if sus_b else None,
                    })

        per_matchup.append({
            "idx": idx,
            "n_actions": len(actions_p1),
            "ally_attack_count": ally_attack_count,
            "ally_frac_enum": float(ally_attack_count / len(actions_p1)) if actions_p1 else 0,
            "ally_attack_nash_weight": float(ally_attack_weight),
            "flagged": flagged[:5],
        })

    weights = np.array([m["ally_attack_nash_weight"] for m in per_matchup])
    fracs = np.array([m["ally_frac_enum"] for m in per_matchup])

    med_weight = float(np.median(weights)) if len(weights) > 0 else 0
    max_weight = float(np.max(weights)) if len(weights) > 0 else 0

    weight_pass = med_weight < 0.05
    max_pass = max_weight < 0.20
    verdict = "PASS" if (weight_pass and max_pass) else "FAIL"

    # Collect all flagged actions for reporting
    all_flagged = []
    for m in per_matchup:
        for f in m["flagged"]:
            all_flagged.append({**f, "matchup_idx": m["idx"]})

    return {
        "check": "11_action_quality", "verdict": verdict,
        "n_matchups": len(per_matchup),
        "ally_attack_nash_weight": {
            "mean": float(weights.mean()) if len(weights) > 0 else None,
            "median": med_weight,
            "max": max_weight,
            "p95": float(np.percentile(weights, 95)) if len(weights) > 0 else None,
        },
        "ally_frac_enum": {
            "mean": float(fracs.mean()) if len(fracs) > 0 else None,
            "median": float(np.median(fracs)) if len(fracs) > 0 else None,
        },
        "flagged_actions": all_flagged[:20],
        "criteria": {
            "median_ally_weight_lt_5pct": weight_pass,
            "max_ally_weight_lt_20pct": max_pass,
        },
    }


# ──────────────────────────────────────────────────────────
# GPU helper — build payoff matrices + BC strategies
# ──────────────────────────────────────────────────────────

def build_payoff_data(dyn_model, bc_model, dataset, indices, device,
                      reward_weights=None) -> list[dict]:
    """Build payoff matrix + BC P1 strategy + action lists for each index."""
    import torch
    from turnone.game.payoff import enumerate_joint_actions, build_payoff_matrix
    from turnone.game.exploitability import bc_strategy_from_logits

    results = []
    for idx in tqdm(indices, desc="Building payoff matrices"):
        example = dataset[idx]
        strat_a = example["strategic_mask_a"].numpy()
        strat_b = example["strategic_mask_b"].numpy()
        opp_strat_a = example["opp_strategic_mask_a"].numpy()
        opp_strat_b = example["opp_strategic_mask_b"].numpy()

        actions_p1 = enumerate_joint_actions(strat_a, strat_b, include_tera=True)
        actions_p2 = enumerate_joint_actions(opp_strat_a, opp_strat_b, include_tera=True)
        if not actions_p1 or not actions_p2:
            continue

        state = {k: example[k] for k in
                 ("team_a", "team_b", "lead_a", "lead_b", "field_state")}

        R = build_payoff_matrix(
            dyn_model, state, actions_p1, actions_p2,
            example["field_state"].numpy(), device,
            reward_weights=reward_weights,
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

        bc_p1 = bc_strategy_from_logits(
            bc_out["logits_a"][0].cpu().numpy(),
            bc_out["logits_b"][0].cpu().numpy(),
            bc_out["logits_tera"][0].cpu().numpy(), actions_p1,
        )

        results.append({
            "idx": int(idx),
            "R": R,
            "bc_p1": bc_p1,
            "actions_p1": actions_p1,
            "actions_p2": actions_p2,
        })

    return results


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

def _print_summary(all_results: dict) -> None:
    print("\n" + "=" * 60)
    print("AUDIT SUMMARY")
    print("=" * 60)
    print(f"  {'Check':<35} {'Verdict':<12} Priority")
    print("  " + "-" * 56)

    order = [
        ("check1",  "1: BC-vs-BC tracking",      "P0"),
        ("check2b", "2b: Stripped BC mass",       "P0"),
        ("check6",  "6: LP verification",         "P0"),
        ("check11", "11: Nash action quality",    "P0"),
        ("check3",  "3: Correlated noise",        "P1"),
        ("check7",  "7: Field ablation",          "P1"),
        ("check8",  "8: Tail analysis",           "P2"),
        ("check10", "10: Team reuse",             "P2"),
    ]
    for key, name, pri in order:
        if key in all_results:
            v = all_results[key]["verdict"]
            print(f"  {name:<35} {v:<12} {pri}")

    p0_keys = ["check1", "check2b", "check6", "check11"]
    p0_fails = [k for k in p0_keys
                if k in all_results and all_results[k]["verdict"] == "FAIL"]
    if p0_fails:
        print(f"\n  ** P0 FAILURES: {p0_fails} -- revisit framing before milestone **")
    elif any(k in all_results for k in p0_keys):
        print(f"\n  All P0 checks passed")


def _save(results: dict, out_dir: Path) -> None:
    def default(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        return str(obj)

    with open(out_dir / "audit_results.json", "w") as f:
        json.dump(results, f, indent=2, default=default)
    print(f"\nResults saved to {out_dir / 'audit_results.json'}")


def _fmt(val, fmt=".4f"):
    """Format a potentially-None value."""
    return f"{val:{fmt}}" if val is not None else "N/A"


def main():
    parser = argparse.ArgumentParser(description="Methodology audit")
    parser.add_argument("--bc_ckpt", required=True)
    parser.add_argument("--dyn_ckpt", required=True)
    parser.add_argument("--test_split", required=True)
    parser.add_argument("--vocab_path", required=True)
    parser.add_argument("--matchup_details", required=True)
    parser.add_argument("--noise_results",
                        default="results/noise_sensitivity/noise_sensitivity.json")
    parser.add_argument("--dynamics_metrics",
                        default="results/dynamics_metrics.json")
    parser.add_argument("--out_dir", default="results/audit/")
    parser.add_argument("--fast_only", action="store_true",
                        help="Only run CPU checks (1, 8, 10)")
    parser.add_argument("--n_check6", type=int, default=50,
                        help="Matchups for LP verification + action quality")
    parser.add_argument("--n_check3", type=int, default=50,
                        help="Matchups for correlated noise experiment")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load matchup details ----
    print("Loading matchup details...")
    matchups = load_matchup_details(args.matchup_details)
    matchup_indices = [m["idx"] for m in matchups]
    stored_by_idx = {m["idx"]: m for m in matchups}
    print(f"  {len(matchups)} matchups")

    # Load noise + dynamics metrics (optional)
    noise_results = {}
    if Path(args.noise_results).exists():
        with open(args.noise_results) as f:
            noise_results = json.load(f)

    reward_mae = 1.192  # fallback
    if Path(args.dynamics_metrics).exists():
        with open(args.dynamics_metrics) as f:
            reward_mae = json.load(f)["reward_mae"]

    all_results = {}

    # ════════════════════════════════════════════════════════
    # FAST CHECKS (CPU only)
    # ════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("FAST CHECKS (CPU)")
    print("=" * 60)
    t_start = time.time()

    # --- Check 1 ---
    print("\n--- Check 1: BC-vs-BC ≈ Nash tracking ---")
    r1 = check1_bc_vs_bc_tracking(matchups)
    all_results["check1"] = r1
    print(f"  beta = {r1['regression']['beta']:.3f}, "
          f"R^2 = {r1['regression']['r_squared']:.3f}")
    print(f"  Top-quintile |gap| = {r1['quintiles'][-1]['gap_abs_mean']:.3f}"
          if r1['quintiles'] else "  No quintiles computed")
    print(f"  Top-50 |gap| = {r1['top50_by_abs_gv']['gap_abs_mean']:.3f}")
    print(f"  -> {r1['verdict']}")

    # --- Check 8 ---
    print("\n--- Check 8: Tail analysis ---")
    r8 = check8_tail_analysis(matchups)
    all_results["check8"] = r8
    print(f"  Full mean = {r8['full_mean']:.3f}, "
          f"trimmed = {r8['trimmed_mean_5pct']:.3f} "
          f"(change {r8['trim_change_pct']:.1f}%)")
    print(f"  Tail >2.0: {r8['tail_gt_2']['n_tail']} matchups "
          f"({r8['tail_gt_2']['pct_tail']:.0%})")
    print(f"  -> {r8['verdict']}")

    # --- Load dataset for Check 10 (and later GPU checks) ---
    print("\nLoading dataset...")
    from turnone.data.dataset import Turn1Dataset, Vocab
    vocab = Vocab.load(args.vocab_path)
    dataset = Turn1Dataset(args.test_split, vocab)
    print(f"  {len(dataset):,} test examples")

    # --- Check 10 ---
    print("\n--- Check 10: Team reuse ---")
    r10 = check10_team_reuse(dataset, matchup_indices)
    all_results["check10"] = r10
    print(f"  Unique: {r10['unique_teams']} / {r10['total_team_slots']} "
          f"(reuse {r10['reuse_rate']:.1%})")
    print(f"  Max appearances: {r10['max_appearances']}, "
          f"teams appearing 2+: {r10['teams_appearing_2plus']}")
    print(f"  -> {r10['verdict']}")

    print(f"\n  Fast checks done in {time.time() - t_start:.1f}s")

    if args.fast_only:
        _print_summary(all_results)
        _save(all_results, out_dir)
        return

    # ════════════════════════════════════════════════════════
    # GPU CHECKS
    # ════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("GPU CHECKS")
    print("=" * 60)

    import torch
    from turnone.models.bc_policy import BCPolicy
    from turnone.models.dynamics import DynamicsModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Loading models...")
    bc_model = BCPolicy.from_checkpoint(args.bc_ckpt, device)
    dyn_model = DynamicsModel.from_checkpoint(args.dyn_ckpt, device)

    # --- Check 2b ---
    print(f"\n--- Check 2b: Stripped BC mass ({len(matchup_indices)} matchups) ---")
    t0 = time.time()
    r2b = check2b_stripped_mass(bc_model, dataset, matchup_indices, device)
    all_results["check2b"] = r2b
    print(f"  Stripped A: median {r2b['stripped_a']['median']:.4f}, "
          f"max {r2b['stripped_a']['max']:.4f}")
    print(f"  Stripped B: median {r2b['stripped_b']['median']:.4f}, "
          f"max {r2b['stripped_b']['max']:.4f}")
    print(f"  Joint total: median {r2b['joint_total']['median']:.4f}, "
          f"min {r2b['joint_total']['min']:.4f}")
    print(f"  -> {r2b['verdict']}  ({time.time() - t0:.1f}s)")

    # --- Build payoff data for Checks 6 + 11 ---
    print(f"\n--- Building {args.n_check6} payoff matrices for Checks 6+11 ---")
    t0 = time.time()
    check6_indices = matchup_indices[:args.n_check6]
    payoff_data_6 = build_payoff_data(
        dyn_model, bc_model, dataset, check6_indices, device)
    print(f"  Built {len(payoff_data_6)} matrices ({time.time() - t0:.1f}s)")

    # --- Check 6 ---
    print("\n--- Check 6: LP verification ---")
    t0 = time.time()
    r6 = check6_lp_verification(payoff_data_6, stored_by_idx)
    all_results["check6"] = r6
    print(f"  Max verification error: {r6['max_error_max']:.2e}")
    print(f"  {r6['pct_within_1e-3']:.0%} within 1e-3, "
          f"{r6['pct_within_1e-2']:.0%} within 1e-2")
    rbs = r6["rebuild_vs_stored"]
    if rbs["max"] is not None:
        print(f"  Rebuild vs stored: max diff {rbs['max']:.2e}, "
              f"mean {rbs['mean']:.2e}")
    print(f"  -> {r6['verdict']}  ({time.time() - t0:.1f}s)")

    # --- Check 11 ---
    print("\n--- Check 11: Nash action quality ---")
    t0 = time.time()
    r11 = check11_action_quality(payoff_data_6, dataset)
    all_results["check11"] = r11
    aw = r11["ally_attack_nash_weight"]
    print(f"  Ally-attack Nash weight: median {_fmt(aw['median'])}, "
          f"max {_fmt(aw['max'])}, p95 {_fmt(aw['p95'])}")
    af = r11["ally_frac_enum"]
    print(f"  Ally-attack fraction of enum: {_fmt(af['mean'])}")
    n_flagged = len(r11["flagged_actions"])
    if n_flagged > 0:
        print(f"  Flagged {n_flagged} actions with >1% Nash weight:")
        for f in r11["flagged_actions"][:5]:
            parts = [x for x in [f.get("sus_a"), f.get("sus_b")] if x]
            print(f"    w={f['nash_weight']:.3f}  {', '.join(parts)}")
    print(f"  -> {r11['verdict']}  ({time.time() - t0:.1f}s)")

    # --- Check 3 ---
    print(f"\n--- Check 3: Correlated noise ({args.n_check3} matchups) ---")
    t0 = time.time()
    noise_rng = np.random.RandomState(args.seed)
    noise_indices = noise_rng.choice(
        len(dataset), size=args.n_check3, replace=False)
    payoff_data_3 = build_payoff_data(
        dyn_model, bc_model, dataset, noise_indices.tolist(), device)
    r3 = check3_correlated_noise(payoff_data_3, reward_mae, noise_results)
    all_results["check3"] = r3
    bs = r3["bias_shift"]
    print(f"  Bias-shift TV median: {_fmt(bs['tv_median'], '.6f')}")
    ct = r3["correlated_1x"]["tv"]
    it = r3["iid_1x"]["tv"]
    print(f"  Correlated TV: {_fmt(ct['mean'])}, I.I.D. TV: {_fmt(it['mean'])}")
    print(f"  Ratio (corr/iid): "
          f"TV={r3['ratios']['tv_corr_over_iid']:.2f}, "
          f"exploit={r3['ratios']['exploit_corr_over_iid']:.2f}")
    print(f"  -> {r3['verdict']}  ({time.time() - t0:.1f}s)")

    # --- Check 7 ---
    print(f"\n--- Check 7: Field ablation (200 matchups) ---")
    t0 = time.time()
    r7 = check7_field_ablation(
        dyn_model, bc_model, dataset, matchup_indices,
        device, stored_by_idx)
    all_results["check7"] = r7
    fm = r7["field_magnitude"]
    print(f"  Reward decomposition (ground truth):")
    print(f"    HP: {fm['hp_mean_abs']:.3f}, KO: {fm['ko_mean_abs']:.3f}, "
          f"Field: {fm['field_mean_abs']:.3f}")
    print(f"    Field fraction: mean {fm['field_fraction_mean']:.3f}, "
          f"median {fm['field_fraction_median']:.3f}")
    ab = r7["ablation"]
    if ab["exploit_change_pct"] is not None:
        print(f"  Ablation (w_field=0): exploit change {ab['exploit_change_pct']:.1f}%")
        print(f"    Baseline: {_fmt(ab['baseline_exploit_mean'], '.3f')}, "
              f"Ablated: {_fmt(ab['ablated_exploit_mean'], '.3f')}")
    print(f"  -> {r7['verdict']}  ({time.time() - t0:.1f}s)")

    # ---- Summary ----
    _print_summary(all_results)
    _save(all_results, out_dir)


if __name__ == "__main__":
    main()
