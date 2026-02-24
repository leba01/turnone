"""Behavioral cloning policy for TurnOne.

Wraps the shared Turn1Encoder and adds:
  - Per-mon action head: shared 2-layer MLP -> 16 logits (masked)
  - Tera head: 3-way classification from lead pair + pooled context

Reference: docs/PROJECT_BIBLE.md Sections 3.3-3.4
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

from turnone.data.action_space import SLOTS_PER_MON
from turnone.models.encoder import EncoderConfig, Turn1Encoder


class BCPolicy(nn.Module):
    """BC policy: encoder + per-mon action heads + tera head.

    Forward returns a dict of masked logits for each output:
      - logits_a: (B, 16) -- action logits for lead A
      - logits_b: (B, 16) -- action logits for lead B
      - logits_tera: (B, 3) -- tera flag logits

    Supports autoregressive mode where logits_b is conditioned on action_a
    via forward_conditioned().
    """

    def __init__(self, vocab_sizes: dict[str, int], cfg: EncoderConfig,
                 autoregressive: bool = False) -> None:
        super().__init__()
        self.cfg = cfg
        self.autoregressive = autoregressive
        self.encoder = Turn1Encoder(vocab_sizes, cfg)

        # Shared per-mon action head (used for both lead A and lead B)
        self.action_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, 16),
        )

        # Tera head: takes concat of lead_A_repr + lead_B_repr + pooled
        self.tera_head = nn.Sequential(
            nn.Linear(cfg.d_model * 3, cfg.d_ff),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, 3),
        )

        # Autoregressive conditioning: embed action_a, add to lead_B repr
        if autoregressive:
            # +1 for padding/no-condition index = SLOTS_PER_MON
            self.action_a_emb = nn.Embedding(SLOTS_PER_MON + 1, cfg.d_model)
            nn.init.normal_(self.action_a_emb.weight, mean=0.0, std=0.02)

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
        mask_a: Tensor,
        mask_b: Tensor,
    ) -> dict[str, Tensor]:
        """Forward pass (independent factorization).

        In autoregressive mode, logits_b is NOT conditioned on action_a here.
        Use forward_conditioned() for the autoregressive logits_b.

        Args:
            team_a: (B, 6, 8) LongTensor.
            team_b: (B, 6, 8) LongTensor.
            lead_a: (B, 2) LongTensor -- lead indices into team_a.
            lead_b: (B, 2) LongTensor -- lead indices into team_b.
            field_state: (B, 5) FloatTensor.
            mask_a: (B, 16) BoolTensor -- valid action slots for lead A.
            mask_b: (B, 16) BoolTensor -- valid action slots for lead B.

        Returns:
            Dict with keys "logits_a", "logits_b", "logits_tera".
        """
        lead_A_repr, lead_B_repr, pooled = self._encode(
            team_a, team_b, lead_a, lead_b, field_state)

        _NEG = -30.0

        logits_a = self.action_head(lead_A_repr)  # (B, 16)
        logits_a = logits_a.masked_fill(~mask_a, _NEG)

        logits_b = self.action_head(lead_B_repr)  # (B, 16)
        logits_b = logits_b.masked_fill(~mask_b, _NEG)

        tera_input = torch.cat([lead_A_repr, lead_B_repr, pooled], dim=-1)
        logits_tera = self.tera_head(tera_input)  # (B, 3)

        return {
            "logits_a": logits_a,
            "logits_b": logits_b,
            "logits_tera": logits_tera,
        }

    def forward_conditioned(
        self,
        team_a: Tensor,
        team_b: Tensor,
        lead_a: Tensor,
        lead_b: Tensor,
        field_state: Tensor,
        mask_a: Tensor,
        mask_b: Tensor,
        action_a: Tensor,
    ) -> dict[str, Tensor]:
        """Forward pass with autoregressive conditioning.

        Predicts logits_a independently, then conditions logits_b on action_a.
        Requires autoregressive=True at init.

        Args:
            action_a: (B,) LongTensor — ground truth or sampled action_a (0-15).

        Returns:
            Dict with "logits_a", "logits_b" (conditioned), "logits_tera".
        """
        assert self.autoregressive, "forward_conditioned requires autoregressive=True"

        lead_A_repr, lead_B_repr, pooled = self._encode(
            team_a, team_b, lead_a, lead_b, field_state)

        _NEG = -30.0

        logits_a = self.action_head(lead_A_repr)
        logits_a = logits_a.masked_fill(~mask_a, _NEG)

        # Condition lead_B representation on action_a
        action_a_emb = self.action_a_emb(action_a)  # (B, d_model)
        lead_B_cond = lead_B_repr + action_a_emb
        logits_b = self.action_head(lead_B_cond)
        logits_b = logits_b.masked_fill(~mask_b, _NEG)

        tera_input = torch.cat([lead_A_repr, lead_B_repr, pooled], dim=-1)
        logits_tera = self.tera_head(tera_input)

        return {
            "logits_a": logits_a,
            "logits_b": logits_b,
            "logits_tera": logits_tera,
        }

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: torch.device) -> BCPolicy:
        """Load a BCPolicy from a training checkpoint.

        Args:
            path: Path to checkpoint file.
            device: Target device.

        Returns:
            Model in eval mode on the given device.
        """
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = EncoderConfig(**ckpt["encoder_config"])
        autoregressive = ckpt.get("autoregressive", False)
        model = cls(ckpt["vocab_sizes"], cfg, autoregressive=autoregressive)
        model.load_state_dict(ckpt["model_state_dict"])
        return model.to(device).eval()
