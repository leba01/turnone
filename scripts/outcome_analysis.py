"""Step 3: Compute summary statistics from battle outcomes and turn-1 rewards."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT_DIR = Path(__file__).resolve().parent.parent / "results/battle_outcomes"


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score interval for a proportion."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p_hat = successes / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = z / denom * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))
    return p_hat, max(0, center - margin), min(1, center + margin)


def rate_with_ci(successes: int, n: int) -> dict:
    mean, lo, hi = wilson_ci(successes, n)
    return {"mean": round(mean, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4), "n": n}


def main():
    # ── Load data ──
    outcomes = []
    with open(OUT_DIR / "outcomes.jsonl") as f:
        for line in f:
            outcomes.append(json.loads(line))

    turn1_data = []
    with open(OUT_DIR / "turn1_vs_outcome.jsonl") as f:
        for line in f:
            turn1_data.append(json.loads(line))

    print(f"Outcomes: {len(outcomes)} battles")
    print(f"Turn-1 joined: {len(turn1_data)} battles")

    # ══════════════════════════════════════════════════════════════════
    # Analysis A: Tera-first win rate
    # ══════════════════════════════════════════════════════════════════

    # Flatten to per-player observations from outcomes data
    # Each battle contributes two player observations
    tera_t1_wins = 0
    tera_t1_total = 0
    tera_later_wins = 0
    tera_later_total = 0
    no_tera_wins = 0
    no_tera_total = 0

    # Head-to-head: one player teras turn 1, other doesn't
    h2h_tera_first_wins = 0
    h2h_total = 0

    # Both tera turn 1
    both_t1_p1_wins = 0
    both_t1_total = 0

    for o in outcomes:
        for side in ("p1", "p2"):
            tera_turn = o[f"{side}_tera_turn"]
            won = o["winner"] == side

            if tera_turn == 1:
                tera_t1_total += 1
                if won:
                    tera_t1_wins += 1
            elif tera_turn is not None:
                tera_later_total += 1
                if won:
                    tera_later_wins += 1
            else:
                no_tera_total += 1
                if won:
                    no_tera_wins += 1

        # Head-to-head: exactly one teras turn 1
        p1_t1 = o["p1_tera_turn"] == 1
        p2_t1 = o["p2_tera_turn"] == 1
        if p1_t1 and not p2_t1:
            h2h_total += 1
            if o["winner"] == "p1":
                h2h_tera_first_wins += 1
        elif p2_t1 and not p1_t1:
            h2h_total += 1
            if o["winner"] == "p2":
                h2h_tera_first_wins += 1

        # Both tera turn 1
        if p1_t1 and p2_t1:
            both_t1_total += 1
            if o["winner"] == "p1":
                both_t1_p1_wins += 1

    total_player_obs = tera_t1_total + tera_later_total + no_tera_total
    overall_tera_t1_rate = tera_t1_total / total_player_obs if total_player_obs else 0

    tera_analysis = {
        "overall_tera_turn1_rate": round(overall_tera_t1_rate, 4),
        "win_rate_tera_turn1": rate_with_ci(tera_t1_wins, tera_t1_total),
        "win_rate_tera_later": rate_with_ci(tera_later_wins, tera_later_total),
        "win_rate_no_tera": rate_with_ci(no_tera_wins, no_tera_total),
        "head_to_head_tera_first_vs_not": {
            "tera_first_wins": round(h2h_tera_first_wins / h2h_total, 4) if h2h_total else None,
            "n": h2h_total,
        },
        "both_tera_turn1_p1_win_rate": round(both_t1_p1_wins / both_t1_total, 4) if both_t1_total else None,
        "both_tera_turn1_n": both_t1_total,
    }

    print("\n═══ Analysis A: Tera-first win rate ═══")
    print(f"Overall tera turn-1 rate: {overall_tera_t1_rate:.1%}")
    print(f"Win rate | tera turn 1: {tera_t1_wins/tera_t1_total:.1%} (n={tera_t1_total})")
    print(f"Win rate | tera later:  {tera_later_wins/tera_later_total:.1%} (n={tera_later_total})")
    print(f"Win rate | no tera:     {no_tera_wins/no_tera_total:.1%} (n={no_tera_total})")
    print(f"Head-to-head (tera T1 vs not): {h2h_tera_first_wins/h2h_total:.1%} (n={h2h_total})")
    print(f"Both tera T1, P1 win rate: {both_t1_p1_wins/both_t1_total:.1%} (n={both_t1_total})")

    # ══════════════════════════════════════════════════════════════════
    # Analysis B: Turn 1 ≠ battle outcome
    # ══════════════════════════════════════════════════════════════════

    rewards = np.array([d["turn1_reward_p1"] for d in turn1_data])
    wins = np.array([1.0 if d["winner"] == "p1" else 0.0 for d in turn1_data])

    # Point-biserial correlation
    pb_r, pb_p = stats.pointbiserialr(wins, rewards)

    # Quintile analysis
    quintile_edges = np.percentile(rewards, [0, 20, 40, 60, 80, 100])
    quintile_win_rates = []
    for q in range(5):
        lo = quintile_edges[q]
        hi = quintile_edges[q + 1]
        if q == 4:
            mask = (rewards >= lo) & (rewards <= hi)
        else:
            mask = (rewards >= lo) & (rewards < hi)
        q_wins = int(wins[mask].sum())
        q_n = int(mask.sum())
        q_rate = q_wins / q_n if q_n else 0
        quintile_win_rates.append({
            "quintile": q + 1,
            "reward_range": [round(float(lo), 3), round(float(hi), 3)],
            "win_rate": round(q_rate, 4),
            "n": q_n,
        })

    turn1_vs_outcome = {
        "point_biserial_r": round(float(pb_r), 4),
        "point_biserial_p": float(pb_p),
        "quintile_win_rates": quintile_win_rates,
        "top_quintile_win_rate": quintile_win_rates[4]["win_rate"],
        "bottom_quintile_win_rate": quintile_win_rates[0]["win_rate"],
    }

    print("\n═══ Analysis B: Turn 1 ≠ battle outcome ═══")
    print(f"Point-biserial r = {pb_r:.4f} (p = {pb_p:.2e})")
    print("Quintile win rates (P1):")
    for q in quintile_win_rates:
        print(f"  Q{q['quintile']} [{q['reward_range'][0]:+.3f}, {q['reward_range'][1]:+.3f}]: "
              f"{q['win_rate']:.1%} (n={q['n']})")

    # ── Save summary ──
    summary = {
        "n_battles_total": len(outcomes),
        "n_battles_with_winner": len(outcomes),  # all loaded have winner
        "n_battles_with_turn1_data": len(turn1_data),
        "tera_analysis": tera_analysis,
        "turn1_vs_outcome": turn1_vs_outcome,
    }

    summary_path = OUT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {summary_path}")


if __name__ == "__main__":
    main()
