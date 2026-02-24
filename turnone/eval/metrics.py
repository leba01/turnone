"""Pure-numpy evaluation metrics for BC policy.

No torch imports. All metric functions operate on numpy arrays.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stable_softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax in float64. Handles -inf correctly (-> 0 prob)."""
    x = logits.astype(np.float64)
    x_max = np.max(x, axis=-1, keepdims=True)
    # If an entire row is -inf, x_max will be -inf; replace with 0 to avoid nan
    x_max = np.where(np.isfinite(x_max), x_max, 0.0)
    e = np.exp(x - x_max)
    return e / e.sum(axis=-1, keepdims=True)


def _per_mon_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | int]:
    """Compute top1, top3, nll, mask_compliance for one mon position.

    Args:
        logits: (N, 16) raw logits; invalid slots should already be -inf.
        labels: (N,) int ground-truth action slot; -1 = fainted before acting.
        mask:   (N, 16) bool; True = valid action slot.

    Returns:
        Dict with keys top1, top3, nll, mask_compliance, n_valid.
    """
    valid = labels != -1
    n_valid = int(valid.sum())

    # Mask compliance is computed over ALL examples (not just valid)
    probs = _stable_softmax(logits)
    masked_prob_sum = (probs * mask).sum(axis=-1)
    compliance = float(masked_prob_sum.mean())

    if n_valid == 0:
        return {
            "top1": 0.0,
            "top3": 0.0,
            "nll": 0.0,
            "mask_compliance": compliance,
            "n_valid": 0,
        }

    v_logits = logits[valid]
    v_labels = labels[valid]
    v_probs = probs[valid]

    # Top-1 accuracy
    preds = np.argmax(v_logits, axis=-1)
    top1 = float((preds == v_labels).mean())

    # Top-3 accuracy
    top3_indices = np.argsort(-v_logits, axis=-1)[:, :3]
    top3 = float(np.any(top3_indices == v_labels[:, None], axis=-1).mean())

    # NLL: -log(p[true_label]), clamped for stability
    true_probs = v_probs[np.arange(len(v_labels)), v_labels]
    true_probs = np.clip(true_probs, 1e-12, 1.0)
    nll = float(-np.log(true_probs).mean())

    return {
        "top1": top1,
        "top3": top3,
        "nll": nll,
        "mask_compliance": compliance,
        "n_valid": n_valid,
    }


# ---------------------------------------------------------------------------
# Main metric function
# ---------------------------------------------------------------------------

def compute_bc_metrics(
    logits_a: np.ndarray,       # (N, 16) float -- raw logits for mon A
    logits_b: np.ndarray,       # (N, 16) float
    logits_tera: np.ndarray,    # (N, 3) float
    action_a: np.ndarray,       # (N,) int -- true action slots, -1 = fainted
    action_b: np.ndarray,       # (N,) int
    tera_label: np.ndarray,     # (N,) int -- 0/1/2
    mask_a: np.ndarray,         # (N, 16) bool
    mask_b: np.ndarray,         # (N, 16) bool
) -> dict[str, float]:
    """Compute all BC evaluation metrics.

    Returns dict with keys:
        top1_a, top1_b, top1_avg   -- per-mon top-1 accuracy (valid only)
        top3_a, top3_b, top3_avg   -- per-mon top-3 accuracy (valid only)
        nll_a, nll_b, nll_avg      -- per-mon NLL (valid only)
        tera_acc                   -- top-1 accuracy on tera flag
        tera_nll                   -- NLL on tera flag
        mask_compliance_a, mask_compliance_b -- prob mass on valid slots
        n_valid_a, n_valid_b       -- count of non-fainted examples
    """
    ma = _per_mon_metrics(logits_a, action_a, mask_a)
    mb = _per_mon_metrics(logits_b, action_b, mask_b)

    # Weighted average for top1/top3/nll (weight by n_valid)
    na, nb = ma["n_valid"], mb["n_valid"]
    total = na + nb
    if total > 0:
        top1_avg = (ma["top1"] * na + mb["top1"] * nb) / total
        top3_avg = (ma["top3"] * na + mb["top3"] * nb) / total
        nll_avg = (ma["nll"] * na + mb["nll"] * nb) / total
    else:
        top1_avg = 0.0
        top3_avg = 0.0
        nll_avg = 0.0

    # Tera metrics (always computed over all examples)
    tera_probs = _stable_softmax(logits_tera)
    tera_preds = np.argmax(logits_tera, axis=-1)
    tera_acc = float((tera_preds == tera_label).mean())

    tera_true_probs = tera_probs[np.arange(len(tera_label)), tera_label]
    tera_true_probs = np.clip(tera_true_probs, 1e-12, 1.0)
    tera_nll = float(-np.log(tera_true_probs).mean())

    return {
        "top1_a": ma["top1"],
        "top1_b": mb["top1"],
        "top1_avg": float(top1_avg),
        "top3_a": ma["top3"],
        "top3_b": mb["top3"],
        "top3_avg": float(top3_avg),
        "nll_a": ma["nll"],
        "nll_b": mb["nll"],
        "nll_avg": float(nll_avg),
        "tera_acc": tera_acc,
        "tera_nll": tera_nll,
        "mask_compliance_a": ma["mask_compliance"],
        "mask_compliance_b": mb["mask_compliance"],
        "n_valid_a": float(na),
        "n_valid_b": float(nb),
    }


# ---------------------------------------------------------------------------
# Stratified variant
# ---------------------------------------------------------------------------

def compute_bc_metrics_stratified(
    logits_a: np.ndarray,
    logits_b: np.ndarray,
    logits_tera: np.ndarray,
    action_a: np.ndarray,
    action_b: np.ndarray,
    tera_label: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Compute BC metrics stratified by subsets.

    Returns a dict of dicts with keys:
        "overall"     -- all examples
        "both_acted"  -- both action_a != -1 and action_b != -1
        "partial"     -- exactly one of action_a/action_b is -1
        "tera_used"   -- tera_label != 0
        "no_tera"     -- tera_label == 0

    Each sub-dict has the same structure as compute_bc_metrics output.
    """
    both = (action_a != -1) & (action_b != -1)
    partial = (action_a == -1) ^ (action_b == -1)
    tera_used = tera_label != 0
    no_tera = tera_label == 0

    subsets: dict[str, np.ndarray] = {
        "overall": np.ones(len(action_a), dtype=bool),
        "both_acted": both,
        "partial": partial,
        "tera_used": tera_used,
        "no_tera": no_tera,
    }

    result: dict[str, dict[str, float]] = {}
    for name, idx_mask in subsets.items():
        n = int(idx_mask.sum())
        if n == 0:
            # Empty subset: return zeros for all keys
            result[name] = {
                "top1_a": 0.0, "top1_b": 0.0, "top1_avg": 0.0,
                "top3_a": 0.0, "top3_b": 0.0, "top3_avg": 0.0,
                "nll_a": 0.0, "nll_b": 0.0, "nll_avg": 0.0,
                "tera_acc": 0.0, "tera_nll": 0.0,
                "mask_compliance_a": 0.0, "mask_compliance_b": 0.0,
                "n_valid_a": 0.0, "n_valid_b": 0.0,
            }
        else:
            result[name] = compute_bc_metrics(
                logits_a[idx_mask],
                logits_b[idx_mask],
                logits_tera[idx_mask],
                action_a[idx_mask],
                action_b[idx_mask],
                tera_label[idx_mask],
                mask_a[idx_mask],
                mask_b[idx_mask],
            )
    return result
