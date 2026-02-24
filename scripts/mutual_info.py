"""Measure empirical mutual information I(action_a; action_b) in training data.

Quantifies how correlated mon A and mon B actions are, bounding the error
from the independence factorization P(a,b,tera) = P(a) * P(b) * P(tera).

Usage:
    python scripts/mutual_info.py --train_split data/assembled/train.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from turnone.data.io_utils import read_jsonl
from turnone.data.action_space import get_target_category, SLOTS_PER_MON


def _entropy(probs: np.ndarray) -> float:
    """Shannon entropy in bits, ignoring zero entries."""
    p = probs[probs > 0]
    return float(-np.sum(p * np.log2(p)))


def compute_mutual_info(
    train_path: str | Path,
) -> dict[str, float]:
    """Compute dataset-wide mutual information between action_a and action_b.

    Only uses examples where both mons acted (both actions non-None).

    Returns dict with: mi_bits, mi_nats, h_joint, h_a, h_b, normalized_mi,
    n_examples, and category breakdowns.
    """
    examples = list(read_jsonl(train_path))

    # Filter to both-acted
    pairs = []
    for ex in examples:
        act = ex["action"]
        if act["action_a"] is not None and act["action_b"] is not None:
            pairs.append((act["action_a"]["slot"], act["action_b"]["slot"]))

    n = len(pairs)
    if n == 0:
        return {"error": "no valid pairs"}

    # Joint distribution P(a, b) as 16x16 histogram
    joint_counts = np.zeros((SLOTS_PER_MON, SLOTS_PER_MON), dtype=np.float64)
    for sa, sb in pairs:
        joint_counts[sa, sb] += 1

    joint_probs = joint_counts / joint_counts.sum()

    # Marginals
    p_a = joint_probs.sum(axis=1)  # (16,)
    p_b = joint_probs.sum(axis=0)  # (16,)

    # Entropies
    h_a = _entropy(p_a)
    h_b = _entropy(p_b)
    h_joint = _entropy(joint_probs.ravel())

    # MI = H(A) + H(B) - H(A,B)
    mi_bits = h_a + h_b - h_joint
    mi_nats = mi_bits * np.log(2)

    # Normalized MI: MI / H(A, B)
    normalized_mi = mi_bits / h_joint if h_joint > 0 else 0.0

    # --- Category breakdown ---
    # For each pair, classify both moves by targeting category
    # We need team data to look up move names
    cat_pairs: Counter[tuple[str, str]] = Counter()
    cat_mi_data: dict[tuple[str, str], list[tuple[int, int]]] = {}

    for ex in examples:
        act = ex["action"]
        if act["action_a"] is None or act["action_b"] is None:
            continue

        sa = act["action_a"]["slot"]
        sb = act["action_b"]["slot"]
        move_a_idx = sa // 4
        move_b_idx = sb // 4

        lead_a_idx = ex["lead_indices_a"][0]
        lead_b_idx = ex["lead_indices_a"][1]
        mon_a = ex["team_a"][lead_a_idx]
        mon_b = ex["team_a"][lead_b_idx]

        cat_a = get_target_category(mon_a["moves"][move_a_idx])
        cat_b = get_target_category(mon_b["moves"][move_b_idx])

        key = (cat_a, cat_b)
        cat_pairs[key] += 1
        if key not in cat_mi_data:
            cat_mi_data[key] = []
        cat_mi_data[key].append((sa, sb))

    # MI per category pair
    category_results = {}
    for key in sorted(cat_mi_data.keys()):
        cat_list = cat_mi_data[key]
        nc = len(cat_list)
        cat_joint = np.zeros((SLOTS_PER_MON, SLOTS_PER_MON), dtype=np.float64)
        for sa, sb in cat_list:
            cat_joint[sa, sb] += 1
        cat_joint_p = cat_joint / cat_joint.sum()
        ca = cat_joint_p.sum(axis=1)
        cb = cat_joint_p.sum(axis=0)
        cat_h_a = _entropy(ca)
        cat_h_b = _entropy(cb)
        cat_h_joint = _entropy(cat_joint_p.ravel())
        cat_mi = cat_h_a + cat_h_b - cat_h_joint
        cat_nmi = cat_mi / cat_h_joint if cat_h_joint > 0 else 0.0
        category_results[f"{key[0]}_{key[1]}"] = {
            "n": nc,
            "frac": nc / n,
            "mi_bits": round(cat_mi, 6),
            "normalized_mi": round(cat_nmi, 6),
        }

    # Top 10 most common (a, b) pairs
    pair_counter = Counter(pairs)
    top_pairs = pair_counter.most_common(10)
    top_pair_mass = sum(c for _, c in top_pairs) / n

    return {
        "n_examples": n,
        "mi_bits": round(mi_bits, 6),
        "mi_nats": round(mi_nats, 6),
        "h_a_bits": round(h_a, 6),
        "h_b_bits": round(h_b, 6),
        "h_joint_bits": round(h_joint, 6),
        "normalized_mi": round(normalized_mi, 6),
        "top_10_pair_mass": round(top_pair_mass, 4),
        "top_10_pairs": [
            {"a": int(a), "b": int(b), "count": int(c), "frac": round(c / n, 4)}
            for (a, b), c in top_pairs
        ],
        "category_breakdown": category_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Measure action mutual information")
    parser.add_argument("--train_split", required=True, help="Path to training JSONL")
    parser.add_argument("--out_path", default=None, help="Output JSON path (optional)")
    args = parser.parse_args()

    results = compute_mutual_info(args.train_split)

    print(f"Examples (both acted): {results['n_examples']:,}")
    print(f"\nMutual Information:")
    print(f"  I(A;B) = {results['mi_bits']:.4f} bits ({results['mi_nats']:.4f} nats)")
    print(f"  H(A) = {results['h_a_bits']:.4f} bits")
    print(f"  H(B) = {results['h_b_bits']:.4f} bits")
    print(f"  H(A,B) = {results['h_joint_bits']:.4f} bits")
    print(f"  Normalized MI = {results['normalized_mi']:.4f} ({results['normalized_mi']*100:.1f}%)")
    print(f"\nTop 10 action pairs cover {results['top_10_pair_mass']*100:.1f}% of mass")

    if results.get("category_breakdown"):
        print(f"\nCategory breakdown:")
        for cat, data in sorted(results["category_breakdown"].items(),
                                key=lambda x: -x[1]["n"]):
            print(f"  {cat:20s}: n={data['n']:6d} ({data['frac']*100:5.1f}%)  "
                  f"MI={data['mi_bits']:.4f} bits  NMI={data['normalized_mi']:.4f}")

    if args.out_path:
        out_path = Path(args.out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
