"""Dynamics model for TurnOne.

Predicts turn-1 outcomes (HP changes, KOs, field state changes) from
state + joint actions.  Uses the shared Turn1Encoder architecture for
state representation but trains its own encoder weights (does NOT share
with BC).  Separate action embeddings for slots (17-way: 0-15 + 16=no-action)
and tera (3-way).  MLP trunk with decomposed output heads:
  - HP head: (B, 4) regression (MSE loss)
  - KO head: (B, 4) binary classification (BCE loss)
  - Weather head: (B, 5) categorical (CE loss, 5 classes)
  - Terrain head: (B, 5) categorical (CE loss, 5 classes)
  - Binary field head: (B, 3) binary [trick_room, tw_p1, tw_p2] (BCE loss)

Reference: docs/PROJECT_BIBLE.md Sections 4.1-4.3
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

from turnone.models.encoder import EncoderConfig, Turn1Encoder

# Index used for fainted-before-acting mons (maps -1 → 16)
NO_ACTION_IDX = 16
NUM_ACTION_SLOTS = 17  # 0-15 valid slots + 16 no-action


@dataclass
class FieldLogits:
    """Decomposed field prediction logits."""
    weather: Tensor   # (B, 5) — 5-class categorical
    terrain: Tensor   # (B, 5) — 5-class categorical
    binary: Tensor    # (B, 3) — [trick_room, tailwind_p1, tailwind_p2]


class DynamicsModel(nn.Module):
    """Predicts turn-1 outcomes from state + joint actions.

    Uses the shared Turn1Encoder architecture for state representation.
    Trains its own encoder weights (does NOT share with BC).
    Action embedding has 17 slots (0-15 = moves, 16 = no-action/fainted).
    MLP trunk -> decomposed output heads for HP, KO, and field state.
    """

    def __init__(
        self,
        vocab_sizes: dict[str, int],
        encoder_cfg: EncoderConfig,
        d_action: int = 32,
        d_hidden: int = 256,
        n_mlp_layers: int = 3,
        dropout: float = 0.1,
        action_cross_attn: bool = False,
        action_attn_heads: int = 2,
        action_attn_layers: int = 1,
    ) -> None:
        """
        Args:
            vocab_sizes: mapping from field name to vocabulary size.
                Required keys: "species", "item", "ability", "tera_type", "move"
            encoder_cfg: encoder hyperparameters (same as BC, separate weights).
            d_action: dimensionality of action/tera embeddings.
            d_hidden: hidden dimension of the MLP trunk.
            n_mlp_layers: number of hidden layers in the MLP trunk.
            dropout: dropout rate in the MLP trunk.
            action_cross_attn: if True, apply Transformer attention over the 4
                mon-action embeddings before concatenation with state.
            action_attn_heads: number of attention heads for action cross-attention.
            action_attn_layers: number of Transformer layers for action cross-attention.
        """
        super().__init__()
        self.encoder_cfg = encoder_cfg
        self.d_action = d_action
        self.d_hidden = d_hidden
        self.n_mlp_layers = n_mlp_layers
        self._dropout = dropout
        self.action_cross_attn = action_cross_attn
        self.action_attn_heads = action_attn_heads
        self.action_attn_layers = action_attn_layers

        # Own encoder (same architecture, separate weights)
        self.encoder = Turn1Encoder(vocab_sizes, encoder_cfg)

        # Action embeddings: 17 slots (0-15 valid + 16 no-action for fainted mons)
        self.emb_action = nn.Embedding(NUM_ACTION_SLOTS, d_action)
        self.emb_tera = nn.Embedding(3, d_action)  # tera flag 0-2

        # Optional: cross-attention over the 4 mon-action tokens
        if action_cross_attn:
            self.action_pos_emb = nn.Embedding(4, d_action)  # our_a, our_b, opp_a, opp_b
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_action,
                nhead=action_attn_heads,
                dim_feedforward=d_action * 4,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.action_attn = nn.TransformerEncoder(
                encoder_layer, num_layers=action_attn_layers,
            )

        # MLP trunk input: pooled encoder (d_model)
        #   + 4 action embeddings (our_a, our_b, opp_a, opp_b slots)
        #   + 2 tera embeddings (our tera, opp tera)
        # When action_cross_attn=True, attended tokens replace raw embeddings
        # in-place, so trunk_in is the same.
        trunk_in = encoder_cfg.d_model + 6 * d_action

        layers: list[nn.Module] = []
        for i in range(n_mlp_layers):
            d_in = trunk_in if i == 0 else d_hidden
            layers.extend([
                nn.Linear(d_in, d_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        self.trunk = nn.Sequential(*layers)

        # Output heads
        self.hp_head = nn.Linear(d_hidden, 4)       # HP delta regression
        self.ko_head = nn.Linear(d_hidden, 4)       # KO binary logits

        # Decomposed field heads (Gap 1 fix: proper loss types for each field component)
        # field_state = [weather_idx(0-4), terrain_idx(0-4), trick_room, tw_p1, tw_p2]
        self.weather_head = nn.Linear(d_hidden, 5)   # 5-class categorical
        self.terrain_head = nn.Linear(d_hidden, 5)   # 5-class categorical
        self.binary_field_head = nn.Linear(d_hidden, 3)  # trick_room, tw_p1, tw_p2

    def encode_state(
        self,
        team_a: Tensor,
        team_b: Tensor,
        lead_a: Tensor,
        lead_b: Tensor,
        field_state: Tensor,
    ) -> Tensor:
        """Encode state once → (B, d_model) pooled representation.

        Use this to avoid redundant encoder calls when the same state is
        paired with many different action combinations (e.g. payoff matrix).
        """
        tokens = self.encoder(team_a, team_b, lead_a, lead_b, field_state)
        return tokens.mean(dim=1)  # (B, d_model)

    def predict_from_pooled(
        self,
        pooled: Tensor,
        action_a: Tensor,
        action_b: Tensor,
        tera_flag: Tensor,
        opp_action_a: Tensor,
        opp_action_b: Tensor,
        opp_tera_flag: Tensor,
    ) -> tuple[Tensor, Tensor, FieldLogits]:
        """Run MLP trunk + heads from pre-computed pooled state.

        Args:
            pooled: (B, d_model) from encode_state().
            action_a .. opp_tera_flag: same as forward().

        Returns:
            Same as forward(): (hp_pred, ko_logits, field_logits).
        """
        # Embed actions
        a_emb_a = self.emb_action(action_a)         # (B, d_action)
        a_emb_b = self.emb_action(action_b)         # (B, d_action)
        t_emb = self.emb_tera(tera_flag)            # (B, d_action)
        oa_emb_a = self.emb_action(opp_action_a)    # (B, d_action)
        oa_emb_b = self.emb_action(opp_action_b)    # (B, d_action)
        ot_emb = self.emb_tera(opp_tera_flag)       # (B, d_action)

        # Optional: cross-attention over the 4 mon-action tokens
        if self.action_cross_attn:
            action_stack = torch.stack(
                [a_emb_a, a_emb_b, oa_emb_a, oa_emb_b], dim=1,
            )  # (B, 4, d_action)
            pos_ids = torch.arange(4, device=action_stack.device)
            action_stack = action_stack + self.action_pos_emb(pos_ids)
            action_stack = self.action_attn(action_stack)  # (B, 4, d_action)
            a_emb_a, a_emb_b, oa_emb_a, oa_emb_b = action_stack.unbind(dim=1)

        trunk_input = torch.cat(
            [pooled, a_emb_a, a_emb_b, t_emb, oa_emb_a, oa_emb_b, ot_emb],
            dim=-1,
        )  # (B, d_model + 6 * d_action)

        h = self.trunk(trunk_input)  # (B, d_hidden)

        hp_pred = self.hp_head(h)      # (B, 4)
        ko_logits = self.ko_head(h)    # (B, 4)

        field_logits = FieldLogits(
            weather=self.weather_head(h),        # (B, 5)
            terrain=self.terrain_head(h),        # (B, 5)
            binary=self.binary_field_head(h),    # (B, 3)
        )

        return hp_pred, ko_logits, field_logits

    def forward(
        self,
        team_a: Tensor,
        team_b: Tensor,
        lead_a: Tensor,
        lead_b: Tensor,
        field_state: Tensor,
        action_a: Tensor,
        action_b: Tensor,
        tera_flag: Tensor,
        opp_action_a: Tensor,
        opp_action_b: Tensor,
        opp_tera_flag: Tensor,
    ) -> tuple[Tensor, Tensor, FieldLogits]:
        """Forward pass.

        State inputs (same as Turn1Encoder):
            team_a: (B, 6, 8) LongTensor -- our team.
            team_b: (B, 6, 8) LongTensor -- opponent team.
            lead_a: (B, 2) LongTensor -- lead indices into team_a.
            lead_b: (B, 2) LongTensor -- lead indices into team_b.
            field_state: (B, 5) FloatTensor -- pre-turn field state.

        Action inputs (already mapped: -1 → 16 by caller):
            action_a: (B,) LongTensor, slot indices 0-16 (our lead A).
            action_b: (B,) LongTensor, slot indices 0-16 (our lead B).
            tera_flag: (B,) LongTensor, 0-2 (our tera).
            opp_action_a: (B,) LongTensor, slot indices 0-16 (opp lead A).
            opp_action_b: (B,) LongTensor, slot indices 0-16 (opp lead B).
            opp_tera_flag: (B,) LongTensor, 0-2 (opp tera).

        Returns:
            hp_pred: (B, 4) predicted HP delta [our_a, our_b, opp_a, opp_b].
                Sign convention: positive = damage taken (matches dataset).
            ko_logits: (B, 4) KO logits [our_a, our_b, opp_a, opp_b].
            field_logits: FieldLogits with weather (B,5), terrain (B,5), binary (B,3).
        """
        pooled = self.encode_state(team_a, team_b, lead_a, lead_b, field_state)
        return self.predict_from_pooled(
            pooled, action_a, action_b, tera_flag,
            opp_action_a, opp_action_b, opp_tera_flag,
        )

    def predict_field_state(self, field_logits: FieldLogits) -> Tensor:
        """Convert field logits to a (B, 5) field state tensor for reward computation.

        Returns:
            (B, 5) tensor: [weather_idx, terrain_idx, trick_room, tw_p1, tw_p2]
            Weather/terrain are argmax indices. Binary fields are sigmoid probabilities.
        """
        weather_idx = field_logits.weather.argmax(dim=-1).float()  # (B,)
        terrain_idx = field_logits.terrain.argmax(dim=-1).float()  # (B,)
        binary_probs = torch.sigmoid(field_logits.binary)          # (B, 3)
        return torch.stack([
            weather_idx,
            terrain_idx,
            binary_probs[:, 0],  # trick_room
            binary_probs[:, 1],  # tailwind_p1
            binary_probs[:, 2],  # tailwind_p2
        ], dim=-1)  # (B, 5)

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: torch.device) -> DynamicsModel:
        """Load a DynamicsModel from a training checkpoint.

        Args:
            path: Path to checkpoint file.
            device: Target device.

        Returns:
            Model in eval mode on the given device.
        """
        ckpt = torch.load(path, map_location=device, weights_only=False)
        encoder_cfg = EncoderConfig(**ckpt["encoder_config"])
        model = cls(
            ckpt["vocab_sizes"],
            encoder_cfg,
            d_action=ckpt.get("d_action", 32),
            d_hidden=ckpt.get("d_hidden", 256),
            n_mlp_layers=ckpt.get("n_mlp_layers", 3),
            dropout=ckpt.get("dropout", 0.1),
            action_cross_attn=ckpt.get("action_cross_attn", False),
            action_attn_heads=ckpt.get("action_attn_heads", 2),
            action_attn_layers=ckpt.get("action_attn_layers", 1),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        return model.to(device).eval()


def remap_actions(actions: Tensor) -> Tensor:
    """Map fainted-mon actions from -1 to NO_ACTION_IDX (16).

    Args:
        actions: (B,) LongTensor with values in {-1, 0, ..., 15}.

    Returns:
        (B,) LongTensor with values in {0, ..., 16}.
    """
    return actions.clamp(min=0).where(actions >= 0, torch.tensor(NO_ACTION_IDX, device=actions.device))
