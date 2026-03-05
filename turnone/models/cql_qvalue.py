"""Conservative Q-Learning (CQL) Q-network for TurnOne.

Factored Q-function: Q(s, slot_a, slot_b, tera) = Q_a(s, slot_a) + Q_b(s, slot_b) + Q_tera(s, tera).

Shares the Turn1Encoder backbone with BCPolicy. Each Q-head is a 2-layer MLP
that outputs per-action Q-values (not logits for classification).

Reference: Kumar et al., "Conservative Q-Learning for Offline RL" (NeurIPS 2020)
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

from turnone.data.action_space import SLOTS_PER_MON, NUM_TERA
from turnone.models.encoder import EncoderConfig, Turn1Encoder


class QLearner(nn.Module):
    """Factored Q-network for horizon-1 CQL.

    Outputs per-action Q-values for each factored component:
      - q_a: (B, 16) — Q-values for lead A's action slots
      - q_b: (B, 16) — Q-values for lead B's action slots
      - q_tera: (B, 3) — Q-values for tera decision
    """

    def __init__(self, vocab_sizes: dict[str, int], cfg: EncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = Turn1Encoder(vocab_sizes, cfg)

        # Per-mon Q-head (shared for lead A and lead B, matching BC's shared head)
        self.q_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, SLOTS_PER_MON),
        )

        # Tera Q-head: takes concat of lead_A_repr + lead_B_repr + pooled
        self.q_head_tera = nn.Sequential(
            nn.Linear(cfg.d_model * 3, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, NUM_TERA),
        )

    def _encode(
        self,
        team_a: Tensor,
        team_b: Tensor,
        lead_a: Tensor,
        lead_b: Tensor,
        field_state: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Shared encoder pass, returns (lead_A_repr, lead_B_repr, pooled)."""
        tokens = self.encoder(team_a, team_b, lead_a, lead_b, field_state)
        B = tokens.size(0)
        batch_idx = torch.arange(B, device=tokens.device)
        lead_A_repr = tokens[batch_idx, lead_a[:, 0]]  # (B, d_model)
        lead_B_repr = tokens[batch_idx, lead_a[:, 1]]  # (B, d_model)
        pooled = tokens.mean(dim=1)  # (B, d_model)
        return lead_A_repr, lead_B_repr, pooled

    def forward(
        self,
        team_a: Tensor,
        team_b: Tensor,
        lead_a: Tensor,
        lead_b: Tensor,
        field_state: Tensor,
    ) -> dict[str, Tensor]:
        """Forward pass producing factored Q-values.

        Returns:
            Dict with keys "q_a" (B,16), "q_b" (B,16), "q_tera" (B,3).
        """
        lead_A_repr, lead_B_repr, pooled = self._encode(
            team_a, team_b, lead_a, lead_b, field_state)

        q_a = self.q_head(lead_A_repr)   # (B, 16)
        q_b = self.q_head(lead_B_repr)   # (B, 16)

        tera_input = torch.cat([lead_A_repr, lead_B_repr, pooled], dim=-1)
        q_tera = self.q_head_tera(tera_input)  # (B, 3)

        return {"q_a": q_a, "q_b": q_b, "q_tera": q_tera}

    @staticmethod
    def extract_policy(
        q_a: Tensor,
        q_b: Tensor,
        q_tera: Tensor,
        valid_actions: list[tuple[int, int, int]],
        tau: float = 1.0,
    ) -> Tensor:
        """Extract a joint policy from factored Q-values via softmax.

        Args:
            q_a: (16,) Q-values for lead A (single example, no batch).
            q_b: (16,) Q-values for lead B.
            q_tera: (3,) Q-values for tera.
            valid_actions: list of (slot_a, slot_b, tera) tuples.
            tau: temperature for softmax.

        Returns:
            (len(valid_actions),) probability vector over joint actions.
        """
        if not valid_actions:
            return torch.zeros(0, device=q_a.device)

        idx_a = torch.tensor([a[0] for a in valid_actions], device=q_a.device)
        idx_b = torch.tensor([a[1] for a in valid_actions], device=q_a.device)
        idx_t = torch.tensor([a[2] for a in valid_actions], device=q_a.device)

        q_joint = q_a[idx_a] + q_b[idx_b] + q_tera[idx_t]
        return torch.softmax(q_joint / tau, dim=0)

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: torch.device) -> QLearner:
        """Load a QLearner from a training checkpoint."""
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = EncoderConfig(**ckpt["encoder_config"])
        model = cls(ckpt["vocab_sizes"], cfg)
        model.load_state_dict(ckpt["model_state_dict"])
        return model.to(device).eval()
