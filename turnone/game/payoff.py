"""Payoff matrix construction for zero-sum game analysis.

Enumerates valid joint actions for both sides and constructs the payoff
matrix by batched dynamics model queries.  Uses cached encoder output to
avoid redundant Transformer computation for the same state.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from turnone.data.action_space import NUM_TERA, SLOTS_PER_MON
from turnone.models.dynamics import DynamicsModel, remap_actions


def enumerate_joint_actions(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    include_tera: bool = True,
) -> list[tuple[int, int, int]]:
    """Enumerate all valid (slot_a, slot_b, tera_flag) triples.

    Args:
        mask_a: (16,) bool — valid action slots for lead A.
        mask_b: (16,) bool — valid action slots for lead B.
        include_tera: if True, enumerate tera_flag in {0, 1, 2}; otherwise only 0.

    Returns:
        List of (slot_a, slot_b, tera_flag) tuples.
    """
    valid_a = np.where(mask_a)[0].tolist()
    valid_b = np.where(mask_b)[0].tolist()
    tera_range = range(NUM_TERA) if include_tera else [0]

    actions = []
    for sa in valid_a:
        for sb in valid_b:
            for tf in tera_range:
                actions.append((sa, sb, tf))
    return actions


def _compute_reward_gpu(
    hp_pred: Tensor,
    ko_logits: Tensor,
    field_pred: Tensor,
    field_before: Tensor,
    w_hp: float = 1.0,
    w_ko: float = 3.0,
    w_field: float = 0.5,
) -> Tensor:
    """Compute zero-sum reward on GPU tensors. Returns (B,) reward tensor.

    Same math as compute_reward_from_dynamics but stays on device.
    """
    # KO probabilities from logits
    ko_probs = torch.sigmoid(ko_logits)

    # HP component: damage dealt to opponents - damage taken by us, normalized
    # Normalize by 200 (max possible = 2 mons × 100 HP each)
    hp_opp = (hp_pred[:, 2] + hp_pred[:, 3]) / 200.0
    hp_ours = (hp_pred[:, 0] + hp_pred[:, 1]) / 200.0
    hp_component = hp_opp - hp_ours

    # KO component
    ko_scored = ko_probs[:, 2] + ko_probs[:, 3]
    ko_suffered = ko_probs[:, 0] + ko_probs[:, 1]
    ko_component = ko_scored - ko_suffered

    # Field component
    tw_before = field_before[:, 3] - field_before[:, 4]
    tw_after = field_pred[:, 3] - field_pred[:, 4]
    tr_before = field_before[:, 2]
    tr_after = field_pred[:, 2]
    field_component = (tw_after - tw_before) + (tr_after - tr_before)

    return w_hp * hp_component + w_ko * ko_component + w_field * field_component


def build_payoff_matrix(
    model: DynamicsModel,
    state: dict[str, Tensor],
    actions_p1: list[tuple[int, int, int]],
    actions_p2: list[tuple[int, int, int]],
    field_before_np: np.ndarray,
    device: torch.device,
    reward_weights: dict[str, float] | None = None,
    batch_size: int = 16384,
) -> np.ndarray:
    """Build |A1| x |A2| payoff matrix via batched dynamics queries.

    Uses encode_state() once, then predict_from_pooled() per batch to avoid
    redundant Transformer encoder computation.  Action tensor construction
    is fully vectorized (no Python per-element loops).

    Args:
        model: DynamicsModel in eval mode.
        state: dict with keys team_a (6,8), team_b (6,8), lead_a (2,),
               lead_b (2,), field_state (5,) — all tensors (unbatched, single example).
        actions_p1: P1's enumerated actions [(slot_a, slot_b, tera_flag), ...].
        actions_p2: P2's enumerated actions [(slot_a, slot_b, tera_flag), ...].
        field_before_np: (5,) numpy array of pre-turn field state.
        device: torch device.
        reward_weights: optional dict with w_hp, w_ko, w_field for reward.
        batch_size: max entries per GPU batch.

    Returns:
        (|A1|, |A2|) payoff matrix (P1's reward).
    """
    if reward_weights is None:
        reward_weights = {"w_hp": 1.0, "w_ko": 3.0, "w_field": 0.5}

    n1 = len(actions_p1)
    n2 = len(actions_p2)
    R = np.zeros((n1, n2), dtype=np.float32)

    # Pre-convert action lists to numpy arrays for vectorized indexing
    p1_arr = np.array(actions_p1, dtype=np.int64)  # (n1, 3)
    p2_arr = np.array(actions_p2, dtype=np.int64)  # (n2, 3)

    # Build all index pairs via meshgrid (vectorized)
    idx_p1, idx_p2 = np.meshgrid(np.arange(n1), np.arange(n2), indexing='ij')
    idx_p1_flat = idx_p1.ravel()  # (n1*n2,)
    idx_p2_flat = idx_p2.ravel()  # (n1*n2,)

    # Encode state once (single example -> batch of 1)
    team_a = state["team_a"].unsqueeze(0).to(device)   # (1, 6, 8)
    team_b = state["team_b"].unsqueeze(0).to(device)   # (1, 6, 8)
    lead_a = state["lead_a"].unsqueeze(0).to(device)   # (1, 2)
    lead_b = state["lead_b"].unsqueeze(0).to(device)   # (1, 2)
    field_state = state["field_state"].unsqueeze(0).to(device)  # (1, 5)

    with torch.no_grad():
        pooled = model.encode_state(team_a, team_b, lead_a, lead_b, field_state)
        # pooled: (1, d_model)

    # Field before on device for GPU reward computation
    field_before_t = torch.from_numpy(field_before_np).unsqueeze(0).to(device)  # (1, 5)

    total_pairs = n1 * n2
    for start in range(0, total_pairs, batch_size):
        end = min(start + batch_size, total_pairs)
        B = end - start

        # Slice index arrays for this batch
        bi1 = idx_p1_flat[start:end]  # (B,)
        bi2 = idx_p2_flat[start:end]  # (B,)

        # Gather action values via numpy fancy indexing (no Python loop)
        p1_batch = p1_arr[bi1]  # (B, 3) — [slot_a, slot_b, tera_flag]
        p2_batch = p2_arr[bi2]  # (B, 3)

        # Convert to device tensors in one shot
        p1_t = torch.from_numpy(p1_batch).to(device)  # (B, 3)
        p2_t = torch.from_numpy(p2_batch).to(device)  # (B, 3)

        p1_slot_a = p1_t[:, 0]
        p1_slot_b = p1_t[:, 1]
        p1_tera = p1_t[:, 2]
        p2_slot_a = p2_t[:, 0]
        p2_slot_b = p2_t[:, 1]
        p2_tera = p2_t[:, 2]

        # Expand cached pooled representation
        pooled_batch = pooled.expand(B, -1)
        field_before_batch = field_before_t.expand(B, -1)

        with torch.no_grad():
            hp_pred, ko_logits, field_logits = model.predict_from_pooled(
                pooled_batch,
                p1_slot_a, p1_slot_b, p1_tera,
                p2_slot_a, p2_slot_b, p2_tera,
            )
            field_pred = model.predict_field_state(field_logits)

            # Compute reward on GPU (no CPU transfer per batch)
            rewards = _compute_reward_gpu(
                hp_pred, ko_logits, field_pred, field_before_batch,
                **reward_weights,
            )

        # Fill payoff matrix vectorized
        R[bi1, bi2] = rewards.float().cpu().numpy()

    return R
