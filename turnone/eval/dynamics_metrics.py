"""Pure-numpy evaluation metrics for dynamics model.

Computes HP MAE/RMSE/R², KO AUC/accuracy/BCE, and field accuracy metrics.
No torch imports.
"""

from __future__ import annotations

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def compute_dynamics_metrics(
    hp_pred: np.ndarray,
    hp_true: np.ndarray,
    ko_logits: np.ndarray,
    ko_true: np.ndarray,
    field_pred: np.ndarray,  # (N, 5) reconstructed field state
    field_true: np.ndarray,  # (N, 5) ground truth field state
) -> dict[str, float]:
    """Compute dynamics model evaluation metrics.

    Args:
        hp_pred: (N, 4) predicted HP deltas.
        hp_true: (N, 4) ground truth HP deltas.
        ko_logits: (N, 4) raw KO logits.
        ko_true: (N, 4) binary KO ground truth.
        field_pred: (N, 5) predicted field state [weather, terrain, tr, tw1, tw2].
        field_true: (N, 5) ground truth field state.

    Returns:
        Dict with keys:
            hp_mae, hp_rmse, hp_r2
            ko_auc, ko_acc, ko_bce
            weather_acc, terrain_acc, binary_field_acc
    """
    # HP metrics
    hp_err = hp_pred - hp_true
    hp_mae = float(np.abs(hp_err).mean())
    hp_rmse = float(np.sqrt((hp_err ** 2).mean()))

    # R² (coefficient of determination)
    ss_res = (hp_err ** 2).sum()
    ss_tot = ((hp_true - hp_true.mean()) ** 2).sum()
    hp_r2 = float(1.0 - ss_res / max(ss_tot, 1e-12))

    # KO metrics
    ko_probs = _sigmoid(ko_logits)
    ko_preds_binary = (ko_probs > 0.5).astype(np.float32)
    ko_acc = float((ko_preds_binary == ko_true).mean())

    # KO BCE (binary cross-entropy)
    eps = 1e-7
    ko_probs_clipped = np.clip(ko_probs, eps, 1.0 - eps)
    ko_bce = float(-(ko_true * np.log(ko_probs_clipped) + (1 - ko_true) * np.log(1 - ko_probs_clipped)).mean())

    # KO AUC (simple per-column average, handling constant columns)
    ko_auc = _compute_auc(ko_true, ko_probs)

    # Field metrics — decomposed
    # Weather: index 0 (categorical 0-4)
    weather_pred = np.round(field_pred[:, 0]).astype(int)
    weather_true = np.round(field_true[:, 0]).astype(int)
    weather_acc = float((weather_pred == weather_true).mean())

    # Terrain: index 1 (categorical 0-4)
    terrain_pred = np.round(field_pred[:, 1]).astype(int)
    terrain_true = np.round(field_true[:, 1]).astype(int)
    terrain_acc = float((terrain_pred == terrain_true).mean())

    # Binary fields: indices 2-4 (trick_room, tw_p1, tw_p2)
    binary_pred = (field_pred[:, 2:5] > 0.5).astype(np.float32)
    binary_true = (field_true[:, 2:5] > 0.5).astype(np.float32)
    binary_field_acc = float((binary_pred == binary_true).mean())

    return {
        "hp_mae": hp_mae,
        "hp_rmse": hp_rmse,
        "hp_r2": hp_r2,
        "ko_auc": ko_auc,
        "ko_acc": ko_acc,
        "ko_bce": ko_bce,
        "weather_acc": weather_acc,
        "terrain_acc": terrain_acc,
        "binary_field_acc": binary_field_acc,
    }


def compute_reward_error(
    hp_pred: np.ndarray,
    hp_true: np.ndarray,
    ko_logits: np.ndarray,
    ko_true: np.ndarray,
    field_pred: np.ndarray,
    field_true: np.ndarray,
    field_before: np.ndarray,
    w_hp: float = 1.0,
    w_ko: float = 3.0,
    w_field: float = 0.5,
) -> dict[str, float]:
    """Compute reward-space error between dynamics predictions and ground truth.

    For observed (state, action_pair), computes ground-truth reward (from actual
    outcomes) and dynamics-predicted reward, then reports error stats.

    Args:
        hp_pred: (N, 4) predicted HP deltas.
        hp_true: (N, 4) ground truth HP deltas.
        ko_logits: (N, 4) raw KO logits from dynamics model.
        ko_true: (N, 4) binary KO ground truth.
        field_pred: (N, 5) predicted post-turn field state.
        field_true: (N, 5) ground truth post-turn field state.
        field_before: (N, 5) pre-turn field state.
        w_hp, w_ko, w_field: reward weights.

    Returns:
        Dict with reward_mae, reward_rmse, reward_correlation, reward_bias.
    """
    from turnone.rl.reward import compute_reward, compute_reward_from_dynamics

    r_true = compute_reward(hp_true, ko_true, field_before, field_true,
                            w_hp=w_hp, w_ko=w_ko, w_field=w_field)
    r_pred = compute_reward_from_dynamics(hp_pred, ko_logits, field_pred, field_before,
                                          w_hp=w_hp, w_ko=w_ko, w_field=w_field)

    err = r_pred - r_true
    reward_mae = float(np.abs(err).mean())
    reward_rmse = float(np.sqrt((err ** 2).mean()))
    reward_bias = float(err.mean())

    # Pearson correlation
    if r_true.std() > 1e-8 and r_pred.std() > 1e-8:
        reward_corr = float(np.corrcoef(r_true, r_pred)[0, 1])
    else:
        reward_corr = 0.0

    return {
        "reward_mae": reward_mae,
        "reward_rmse": reward_rmse,
        "reward_correlation": reward_corr,
        "reward_bias": reward_bias,
    }


def _compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute average AUC across columns. Skip columns with only one class."""
    aucs = []
    for col in range(y_true.shape[1]):
        yt = y_true[:, col]
        ys = y_score[:, col]
        # Skip if only one class
        if yt.min() == yt.max():
            continue
        # Simple AUC via sorted pairs
        aucs.append(_auc_single(yt, ys))
    return float(np.mean(aucs)) if aucs else 0.0


def _auc_single(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute AUC-ROC for a single binary column via the Wilcoxon-Mann-Whitney statistic."""
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    # Count concordant pairs
    n_concordant = 0
    n_tied = 0
    for p in pos:
        n_concordant += (neg < p).sum()
        n_tied += (neg == p).sum()
    return float((n_concordant + 0.5 * n_tied) / (len(pos) * len(neg)))
