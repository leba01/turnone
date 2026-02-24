"""Shared Transformer encoder for TurnOne.

Adapts TurnZero's OTSTransformer into a headless encoder that produces
per-token representations (B, 13, d_model) for downstream consumption by
the BC policy heads and the dynamics model.

Key differences from TurnZero:
  - Lead position embedding (3-way: not-lead / lead-A / lead-B)
  - Field state token (5-dim projected to d_model) as a 13th token
  - No pooling or classification head -- returns full token tensor

Reference: docs/PROJECT_BIBLE.md Sections 3.1-3.2
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class EncoderConfig:
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 512
    dropout: float = 0.1


class Turn1Encoder(nn.Module):
    """Permutation-equivariant Transformer encoder over 12 mon tokens + 1 field token.

    Input: two teams (B,6,8), lead indices for each side (B,2), field state (B,5).
    Output: (B, 13, d_model) contextualized representations.
    """

    NUM_MONS: int = 12  # 6 per side
    NUM_FIELDS: int = 8  # per-mon integer fields

    def __init__(self, vocab_sizes: dict[str, int], cfg: EncoderConfig) -> None:
        """
        Args:
            vocab_sizes: mapping from field name to vocabulary size.
                Required keys: "species", "item", "ability", "tera_type", "move"
            cfg: encoder hyperparameters.
        """
        super().__init__()
        self.cfg = cfg

        # --- Embedding tables (one per field type) ---
        self.emb_species = nn.Embedding(vocab_sizes["species"], cfg.d_model)
        self.emb_item = nn.Embedding(vocab_sizes["item"], cfg.d_model)
        self.emb_ability = nn.Embedding(vocab_sizes["ability"], cfg.d_model)
        self.emb_tera = nn.Embedding(vocab_sizes["tera_type"], cfg.d_model)
        self.emb_move = nn.Embedding(vocab_sizes["move"], cfg.d_model)  # shared for 4 moves

        # Side embedding: 0=team_a, 1=team_b
        self.emb_side = nn.Embedding(2, cfg.d_model)

        # Lead embedding: 0=not lead, 1=lead_A, 2=lead_B
        self.emb_lead = nn.Embedding(3, cfg.d_model)

        # --- Field state projection (5-dim -> d_model) ---
        self.field_proj = nn.Linear(5, cfg.d_model)

        # --- Pre-norm Transformer encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.n_layers,
            norm=nn.LayerNorm(cfg.d_model),  # final norm after last layer
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize embeddings with normal_(mean=0, std=0.02)."""
        init_std = 0.02
        for emb in [
            self.emb_species, self.emb_item, self.emb_ability,
            self.emb_tera, self.emb_move, self.emb_side, self.emb_lead,
        ]:
            nn.init.normal_(emb.weight, mean=0.0, std=init_std)

    def _embed_team(self, team: Tensor, side: int, lead_flags: Tensor) -> Tensor:
        """Embed a team of 6 mons into (B, 6, d_model).

        Args:
            team: (B, 6, 8) LongTensor -- fields per mon:
                  [species, item, ability, tera_type, move0, move1, move2, move3]
            side: 0 for team_a, 1 for team_b.
            lead_flags: (B, 6) LongTensor -- 0/1/2 per position.
        """
        e = (
            self.emb_species(team[:, :, 0])
            + self.emb_item(team[:, :, 1])
            + self.emb_ability(team[:, :, 2])
            + self.emb_tera(team[:, :, 3])
            + self.emb_move(team[:, :, 4])
            + self.emb_move(team[:, :, 5])
            + self.emb_move(team[:, :, 6])
            + self.emb_move(team[:, :, 7])
        )
        # Side embedding: broadcast (1, d_model) -> (B, 6, d_model)
        side_idx = torch.tensor(side, device=team.device, dtype=torch.long)
        e = e + self.emb_side(side_idx)
        # Lead embedding: (B, 6) -> (B, 6, d_model)
        e = e + self.emb_lead(lead_flags)
        return e

    @staticmethod
    def _build_lead_flags(lead_indices: Tensor, num_mons: int = 6) -> Tensor:
        """Build (B, 6) lead flag tensor from (B, 2) lead indices.

        Args:
            lead_indices: (B, 2) LongTensor of positions 0-5.
            num_mons: number of mons per team (always 6).

        Returns:
            (B, 6) LongTensor with values 0=not-lead, 1=lead_A, 2=lead_B.
        """
        B = lead_indices.size(0)
        flags = torch.zeros(B, num_mons, device=lead_indices.device, dtype=torch.long)
        batch_idx = torch.arange(B, device=lead_indices.device)
        flags[batch_idx, lead_indices[:, 0]] = 1  # lead A
        flags[batch_idx, lead_indices[:, 1]] = 2  # lead B
        return flags

    def forward(
        self,
        team_a: Tensor,
        team_b: Tensor,
        lead_a: Tensor,
        lead_b: Tensor,
        field_state: Tensor,
    ) -> Tensor:
        """Forward pass.

        Args:
            team_a: (B, 6, 8) LongTensor -- our team.
            team_b: (B, 6, 8) LongTensor -- opponent team.
            lead_a: (B, 2) LongTensor -- lead indices into team_a.
            lead_b: (B, 2) LongTensor -- lead indices into team_b.
            field_state: (B, 5) FloatTensor -- pre-turn field state.

        Returns:
            (B, 13, d_model) contextualized token representations.
            Tokens 0-5: team_a mons, tokens 6-11: team_b mons, token 12: field.
        """
        # Build lead flags for each team
        lead_flags_a = self._build_lead_flags(lead_a)   # (B, 6)
        lead_flags_b = self._build_lead_flags(lead_b)   # (B, 6)

        # Embed both teams -> (B, 6, d_model) each
        emb_a = self._embed_team(team_a, side=0, lead_flags=lead_flags_a)
        emb_b = self._embed_team(team_b, side=1, lead_flags=lead_flags_b)

        # Concatenate mon tokens -> (B, 12, d_model)
        tokens = torch.cat([emb_a, emb_b], dim=1)

        # Field token -> (B, 1, d_model)
        field_token = self.field_proj(field_state).unsqueeze(1)

        # Full sequence -> (B, 13, d_model)
        tokens = torch.cat([tokens, field_token], dim=1)

        # Transformer encoder
        tokens = self.encoder(tokens)

        return tokens
