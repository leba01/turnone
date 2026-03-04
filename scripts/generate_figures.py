"""Generate 3 publication-quality figures for the final paper.

Each figure answers one question:
1. SV decay — "Why is convention free?" (the game is 3-dimensional)
2. Tera + aggression shift — "Where does convention fail?"
3. Smoothed BR convergence — "How far is convention from Nash?"

Usage:
    python scripts/generate_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)

OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)


def fig1_sv_decay():
    """Cumulative energy vs rank — why is convention free?"""
    with open("results/phase2/svd.json") as f:
        data = json.load(f)

    ks = [1, 2, 3, 5, 10]
    means = [data["aggregate"]["top_k_frac_means"][f"top_{k}"] for k in ks]

    per = data["per_matchup"]
    all_fracs = {k: [m[f"top_{k}_frac"] for m in per] for k in ks}

    fig, ax = plt.subplots(figsize=(3.4, 2.4))

    lo = [np.percentile(all_fracs[k], 5) for k in ks]
    hi = [np.percentile(all_fracs[k], 95) for k in ks]
    ax.fill_between(ks, lo, hi, alpha=0.2, color="C0", label="5th--95th pctile")

    ax.plot(ks, means, "o-", color="C0", markersize=4, linewidth=1.5, label="Mean")

    ax.axhline(0.96, color="gray", linestyle="--", linewidth=0.7, alpha=0.7)
    ax.annotate(
        "96% at $k{=}3$",
        xy=(3, 0.96),
        xytext=(5.5, 0.91),
        fontsize=8,
        ha="center",
        arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
    )

    ax.set_xlabel("Number of singular values ($k$)")
    ax.set_ylabel("Cumulative energy fraction")
    ax.set_ylim(0.55, 1.02)
    ax.set_xticks(ks)
    ax.legend(loc="lower right", frameon=True, framealpha=0.9)
    ax.grid(True, alpha=0.3, linewidth=0.5)

    fig.savefig(OUT_DIR / "fig1_sv_decay.pdf")
    plt.close(fig)
    print("  fig1_sv_decay.pdf")


def fig2_tera_and_aggression():
    """BC vs BR behavioral composition — tera and targeting."""
    with open("results/tera_decomposition/tera_decomposition.json") as f:
        tera_data = json.load(f)
    with open("results/exploitation/analysis.json") as f:
        exploit_data = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.4))

    # Panel A: Tera usage
    bc_tera = tera_data["aggregate"]["summary"]["bc_uses_tera_mean"]
    br_tera = tera_data["aggregate"]["summary"]["br_uses_tera_frac"]

    bars = ax1.bar(
        ["BC (Expert)", "BR (Optimal)"],
        [bc_tera * 100, br_tera * 100],
        color=["C0", "C3"],
        alpha=0.85,
        width=0.5,
    )
    ax1.set_ylabel("Tera usage (%)")
    ax1.set_ylim(0, 110)
    for bar, val in zip(bars, [bc_tera, br_tera]):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{val * 100:.0f}%",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )
    ax1.set_title("(a) Terastallization", fontsize=10)
    ax1.grid(True, axis="y", alpha=0.3, linewidth=0.5)

    # Panel B: Aggression (joint patterns)
    direction = exploit_data["systematic_direction"]

    bc_pats = direction["joint_patterns"]["bc_mean"]
    br_pats = direction["joint_patterns"]["br_frac"]

    patterns = ["both_offensive", "mixed", "both_support"]
    labels = ["Both\noffensive", "Mixed", "Both\nsupport"]
    bc_vals = [bc_pats.get(p, 0) * 100 for p in patterns]
    br_vals = [br_pats.get(p, 0) * 100 for p in patterns]

    x = np.arange(len(patterns))
    w = 0.35
    ax2.bar(x - w / 2, bc_vals, w, label="BC (Expert)", color="C0", alpha=0.85)
    ax2.bar(x + w / 2, br_vals, w, label="BR (Optimal)", color="C3", alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("Frequency (%)")
    ax2.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=7)
    ax2.set_title("(b) Joint targeting pattern", fontsize=10)
    ax2.grid(True, axis="y", alpha=0.3, linewidth=0.5)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig2_tera_aggression.pdf")
    plt.close(fig)
    print("  fig2_tera_aggression.pdf")


def fig3_smoothed_br():
    """Smoothed BR convergence — how far is convention from Nash?"""
    with open("results/smoothed_br/smoothed_br.json") as f:
        data = json.load(f)

    best_cfg = data["aggregated"]["beta=0.01_eta=0.1"]

    mt = best_cfg.get("mean_trajectory", {})
    traj = mt.get("exploit", []) if isinstance(mt, dict) else []
    p25_mt = best_cfg.get("p25_trajectory", {})
    p75_mt = best_cfg.get("p75_trajectory", {})
    p25 = p25_mt.get("exploit") if isinstance(p25_mt, dict) else None
    p75 = p75_mt.get("exploit") if isinstance(p75_mt, dict) else None

    fig, ax = plt.subplots(figsize=(3.4, 2.4))

    if traj and len(traj) > 0:
        iters = np.arange(len(traj))
        step = max(1, len(traj) // 100)
        idx = np.arange(0, len(traj), step)

        ax.plot(iters[idx], np.array(traj)[idx], color="C0", linewidth=1.5)
        if p25 and p75:
            ax.fill_between(
                iters[idx],
                np.array(p25)[idx],
                np.array(p75)[idx],
                alpha=0.2,
                color="C0",
                label="IQR",
            )

        # Mark the 19-iteration point (exploit < 0.5)
        if len(traj) > 19:
            ax.axvline(19, color="C3", linestyle=":", linewidth=0.8, alpha=0.7)
            ax.annotate(
                f"iter 19: {traj[19]:.2f}",
                xy=(19, traj[19]),
                xytext=(160, traj[19] + 0.5),
                fontsize=7,
                arrowprops=dict(arrowstyle="->", color="C3", lw=0.8),
            )

    ax.set_xlabel("Smoothed BR iteration")
    ax.set_ylabel("Mean exploitability")
    ax.grid(True, alpha=0.3, linewidth=0.5)

    fig.savefig(OUT_DIR / "fig3_smoothed_br.pdf")
    plt.close(fig)
    print("  fig3_smoothed_br.pdf")


def main():
    print("Generating figures...")
    fig1_sv_decay()
    fig2_tera_and_aggression()
    fig3_smoothed_br()
    print(f"\nAll 3 figures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
