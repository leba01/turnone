"""Zero-sum reward function for turn-1 outcomes.

Reward is computed from P1's perspective. P2's reward = -P1's reward.
Higher reward = better for P1.

Components:
  - HP advantage: damage dealt to opponents minus damage taken
  - KO advantage: KOs scored minus KOs suffered (heavily weighted)
  - Field advantage: net change in field control (tailwind, terrain, etc.)
"""

from __future__ import annotations

import numpy as np


def compute_reward(
    hp_delta: np.ndarray,
    ko_flags: np.ndarray,
    field_before: np.ndarray,
    field_after: np.ndarray,
    w_hp: float = 1.0,
    w_ko: float = 3.0,
    w_field: float = 0.5,
) -> np.ndarray:
    """Compute zero-sum reward from P1's perspective using ground-truth outcomes.

    Args:
        hp_delta: (N, 4) HP lost [our_a, our_b, opp_a, opp_b]. Positive = damage taken.
        ko_flags: (N, 4) binary KO indicators [our_a, our_b, opp_a, opp_b].
        field_before: (N, 5) pre-turn field state.
        field_after: (N, 5) post-turn field state.
        w_hp: weight for HP component.
        w_ko: weight for KO component.
        w_field: weight for field component.

    Returns:
        (N,) reward array. Positive = good for P1.
    """
    # HP component: damage dealt to opponents - damage taken by us
    # hp_delta is "HP lost", so opponent damage = opp_a + opp_b, our damage = our_a + our_b
    # Normalize by 200 (max possible = 2 mons × 100 HP each)
    hp_opp = (hp_delta[:, 2] + hp_delta[:, 3]) / 200.0  # damage we dealt
    hp_ours = (hp_delta[:, 0] + hp_delta[:, 1]) / 200.0  # damage we took
    hp_component = hp_opp - hp_ours  # positive = we dealt more damage

    # KO component: KOs scored - KOs suffered
    ko_scored = ko_flags[:, 2] + ko_flags[:, 3]    # opponent KOs (we scored)
    ko_suffered = ko_flags[:, 0] + ko_flags[:, 1]  # our KOs (we suffered)
    ko_component = ko_scored - ko_suffered

    # Field component: improvement in field control
    # Field state: [weather_idx, terrain_idx, trick_room, tailwind_ours, tailwind_opp]
    # We care about tailwind advantage (index 3 vs 4) and trick room (index 2)
    # Simple: net gain in our tailwind minus opponent tailwind
    tw_before = field_before[:, 3] - field_before[:, 4]  # our tailwind advantage before
    tw_after = field_after[:, 3] - field_after[:, 4]      # our tailwind advantage after
    tr_before = field_before[:, 2]
    tr_after = field_after[:, 2]
    # Trick Room change (setting it is valuable, getting it set against you is bad)
    # For simplicity: field advantage = tailwind improvement + trick room change
    field_component = (tw_after - tw_before) + (tr_after - tr_before)

    return w_hp * hp_component + w_ko * ko_component + w_field * field_component


def compute_reward_from_dynamics(
    hp_pred: np.ndarray,
    ko_logits: np.ndarray,
    field_pred: np.ndarray,
    field_before: np.ndarray,
    w_hp: float = 1.0,
    w_ko: float = 3.0,
    w_field: float = 0.5,
) -> np.ndarray:
    """Compute reward using dynamics model outputs.

    Same as compute_reward but uses model predictions:
    - hp_pred: (N, 4) predicted HP delta (used directly)
    - ko_logits: (N, 4) KO logits -> converted to probabilities via sigmoid
    - field_pred: (N, 5) predicted post-turn field state
    - field_before: (N, 5) pre-turn field state (ground truth)
    """
    # Use sigmoid to convert ko_logits to probabilities
    ko_probs = 1.0 / (1.0 + np.exp(-np.clip(ko_logits, -30, 30)))

    return compute_reward(
        hp_delta=hp_pred,
        ko_flags=ko_probs,
        field_before=field_before,
        field_after=field_pred,
        w_hp=w_hp,
        w_ko=w_ko,
        w_field=w_field,
    )
