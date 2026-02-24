"""PyTorch Dataset for TurnOne turn-1 examples.

Each example encodes:
  - State: team_a (6,8), team_b (6,8), lead position flags
  - Field state: weather, terrain, trick room (from switch-in abilities)
  - Action labels: per-mon action slots (0-15) + tera flag (0-2)
  - Masks: valid action slots per mon (16-dim bool)
  - Resolution: HP changes, KO flags, field state changes

The Vocab class is reused from TurnZero's design: index 0 = <UNK>.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from turnone.data.action_space import (
    compute_action_mask,
    compute_strategic_mask,
    SLOTS_PER_MON,
    NUM_TERA,
    get_target_category,
)
from turnone.data.io_utils import read_jsonl


# Field names for the 8-int per-mon encoding (same as TurnZero)
_FIELD_TYPES = ("species", "item", "ability", "tera_type", "move")

# Weather/terrain vocabularies for encoding field state
WEATHER_VOCAB = {"none": 0, "RainDance": 1, "SunnyDay": 2, "Sandstorm": 3, "Snow": 4,
                  "Snowscape": 4, "DesolateLand": 2}  # aliases
TERRAIN_VOCAB = {
    "none": 0,
    "Electric Terrain": 1,
    "Grassy Terrain": 2,
    "Misty Terrain": 3,
    "Psychic Terrain": 4,
}


class Vocab:
    """Token-to-index mappings for categorical OTS fields.

    Index 0 is always <UNK>. Built from training-split examples only.
    """

    UNK = "<UNK>"
    UNK_IDX = 0

    def __init__(self) -> None:
        self._tok2idx: dict[str, dict[str, int]] = {}
        for ft in _FIELD_TYPES:
            self._tok2idx[ft] = {self.UNK: self.UNK_IDX}

    @classmethod
    def from_examples(cls, examples: list[dict[str, Any]]) -> Vocab:
        """Build vocab from training examples."""
        vocab = cls()
        token_sets: dict[str, set[str]] = defaultdict(set)

        for ex in examples:
            for team_key in ("team_a", "team_b"):
                for mon in ex[team_key]:
                    token_sets["species"].add(mon["species"])
                    token_sets["item"].add(mon["item"])
                    token_sets["ability"].add(mon["ability"])
                    token_sets["tera_type"].add(mon["tera_type"])
                    for m in mon["moves"]:
                        token_sets["move"].add(m)

        for ft in _FIELD_TYPES:
            tokens_sorted = sorted(token_sets[ft] - {cls.UNK, "UNK"})
            for tok in tokens_sorted:
                vocab._tok2idx[ft][tok] = len(vocab._tok2idx[ft])

        return vocab

    def encode(self, field_type: str, token: str) -> int:
        return self._tok2idx[field_type].get(token, self.UNK_IDX)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._tok2idx, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> Vocab:
        vocab = cls()
        with open(path) as f:
            vocab._tok2idx = json.load(f)
        return vocab

    @property
    def vocab_sizes(self) -> dict[str, int]:
        return {ft: len(mapping) for ft, mapping in self._tok2idx.items()}

    def __repr__(self) -> str:
        sizes = self.vocab_sizes
        return f"Vocab({', '.join(f'{k}={v}' for k, v in sizes.items())})"


def _encode_team(team_list: list[dict], vocab: Vocab) -> torch.Tensor:
    """Encode a team (list of 6 mon dicts) as a (6, 8) LongTensor."""
    rows = []
    for mon in team_list:
        row = [
            vocab.encode("species", mon["species"]),
            vocab.encode("item", mon["item"]),
            vocab.encode("ability", mon["ability"]),
            vocab.encode("tera_type", mon["tera_type"]),
            vocab.encode("move", mon["moves"][0]),
            vocab.encode("move", mon["moves"][1]),
            vocab.encode("move", mon["moves"][2]),
            vocab.encode("move", mon["moves"][3]),
        ]
        rows.append(row)
    return torch.tensor(rows, dtype=torch.long)


def _encode_field_state(field_dict: dict) -> torch.Tensor:
    """Encode field state as a (5,) float tensor.

    [weather_idx, terrain_idx, trick_room, tailwind_ours, tailwind_opp]
    """
    weather = field_dict.get("weather") or "none"
    terrain = field_dict.get("terrain") or "none"
    return torch.tensor([
        WEATHER_VOCAB.get(weather, 0),
        TERRAIN_VOCAB.get(terrain, 0),
        float(field_dict.get("trick_room", False)),
        float(field_dict.get("tailwind_p1", False)),  # ours in directed
        float(field_dict.get("tailwind_p2", False)),   # opp in directed
    ], dtype=torch.float)


def _compute_mask_for_lead(team_list: list[dict], lead_idx: int) -> np.ndarray:
    """Compute 16-slot action mask for one lead."""
    mon = team_list[lead_idx]
    return compute_action_mask(mon["moves"])


def _compute_strategic_mask_for_lead(team_list: list[dict], lead_idx: int) -> np.ndarray:
    """Compute 16-slot strategic action mask for one lead.

    Like _compute_mask_for_lead but excludes target=3 for single-target moves.
    """
    mon = team_list[lead_idx]
    return compute_strategic_mask(mon["moves"])


def _canonicalize_action_slot(slot: int, team_list: list[dict], lead_idx: int) -> int:
    """Remap target=3 → target=0 for single-target moves.

    This removes the protocol artifact where failed/redirected single-target
    moves show as self-targeting.  Self-target and spread moves are unchanged.

    Args:
        slot: action slot (0-15) or -1 (fainted).
        team_list: raw team data (list of 6 mon dicts).
        lead_idx: which mon in the team is the active lead.

    Returns:
        Canonicalized slot.
    """
    if slot < 0:
        return slot
    move_idx = slot // 4
    target = slot % 4
    if target == 3:
        move_name = team_list[lead_idx]["moves"][move_idx]
        if get_target_category(move_name) == "single":
            return move_idx * 4 + 0  # remap to opp_A
    return slot


class Turn1Dataset(Dataset):
    """PyTorch dataset for turn-1 examples.

    Loads all examples into memory and produces:
    - team_a, team_b: (6, 8) encoded teams
    - lead_mask: (2,) lead positions in team_a [0-5, 0-5]
    - field_state: (5,) pre-turn field state
    - action_a, action_b: int action slot labels (0-15), -1 if unobserved
    - mask_a, mask_b: (16,) bool valid action masks
    - tera_label: int (0-2)
    - opp_action_a, opp_action_b: opponent action labels
    - opp_mask_a, opp_mask_b: opponent action masks
    - opp_tera_label: opponent tera label
    - resolution: HP and field outcome data
    """

    def __init__(self, jsonl_path: str | Path, vocab: Vocab,
                 require_both_actions: bool = False,
                 canonicalize_targets: bool = False) -> None:
        self.vocab = vocab
        self.canonicalize_targets = canonicalize_targets
        self.examples: list[dict[str, Any]] = list(read_jsonl(jsonl_path))
        if require_both_actions:
            self.examples = [
                ex for ex in self.examples
                if (ex["action"]["action_a"] is not None
                    and ex["action"]["action_b"] is not None)
            ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ex = self.examples[idx]

        team_a = _encode_team(ex["team_a"], self.vocab)
        team_b = _encode_team(ex["team_b"], self.vocab)

        lead_a = torch.tensor(ex["lead_indices_a"], dtype=torch.long)
        lead_b = torch.tensor(ex["lead_indices_b"], dtype=torch.long)

        field_state = _encode_field_state(ex["pre_turn_field"])

        # Action labels (our side)
        action = ex["action"]
        action_a_slot = action["action_a"]["slot"] if action["action_a"] else -1
        action_b_slot = action["action_b"]["slot"] if action["action_b"] else -1
        tera_label = action["tera_flag"]

        # Opponent actions (extract early so canonicalization can use them)
        opp = ex["opponent_action"]
        opp_a_slot = opp["action_a"]["slot"] if opp["action_a"] else -1
        opp_b_slot = opp["action_b"]["slot"] if opp["action_b"] else -1
        opp_tera = opp["tera_flag"]

        # Canonicalize targets: remap target=3 → target=0 for single-target moves
        if self.canonicalize_targets:
            action_a_slot = _canonicalize_action_slot(
                action_a_slot, ex["team_a"], ex["lead_indices_a"][0])
            action_b_slot = _canonicalize_action_slot(
                action_b_slot, ex["team_a"], ex["lead_indices_a"][1])
            opp_a_slot = _canonicalize_action_slot(
                opp_a_slot, ex["team_b"], ex["lead_indices_b"][0])
            opp_b_slot = _canonicalize_action_slot(
                opp_b_slot, ex["team_b"], ex["lead_indices_b"][1])

        # Action masks (our side)
        mask_a = _compute_mask_for_lead(ex["team_a"], ex["lead_indices_a"][0])
        mask_b = _compute_mask_for_lead(ex["team_a"], ex["lead_indices_a"][1])

        # Opponent masks
        opp_mask_a = _compute_mask_for_lead(ex["team_b"], ex["lead_indices_b"][0])
        opp_mask_b = _compute_mask_for_lead(ex["team_b"], ex["lead_indices_b"][1])

        # Strategic masks (exclude target=3 for single-target moves)
        strat_mask_a = _compute_strategic_mask_for_lead(ex["team_a"], ex["lead_indices_a"][0])
        strat_mask_b = _compute_strategic_mask_for_lead(ex["team_a"], ex["lead_indices_a"][1])
        opp_strat_mask_a = _compute_strategic_mask_for_lead(ex["team_b"], ex["lead_indices_b"][0])
        opp_strat_mask_b = _compute_strategic_mask_for_lead(ex["team_b"], ex["lead_indices_b"][1])

        # Resolution (for dynamics model training)
        res = ex["resolution"]
        hp_delta = torch.tensor([
            res["hp_before"]["our_a"] - res["hp_after"]["our_a"],
            res["hp_before"]["our_b"] - res["hp_after"]["our_b"],
            res["hp_before"]["opp_a"] - res["hp_after"]["opp_a"],
            res["hp_before"]["opp_b"] - res["hp_after"]["opp_b"],
        ], dtype=torch.float)  # HP lost (positive = damage taken)

        ko_flags = torch.tensor([
            1.0 if "our_a" in res["kos"] else 0.0,
            1.0 if "our_b" in res["kos"] else 0.0,
            1.0 if "opp_a" in res["kos"] else 0.0,
            1.0 if "opp_b" in res["kos"] else 0.0,
        ], dtype=torch.float)

        field_after = _encode_field_state(res["field_state"])

        return {
            "team_a": team_a,             # (6, 8)
            "team_b": team_b,             # (6, 8)
            "lead_a": lead_a,             # (2,) indices into team_a
            "lead_b": lead_b,             # (2,) indices into team_b
            "field_state": field_state,    # (5,)
            "action_a": action_a_slot,     # int 0-15 or -1
            "action_b": action_b_slot,     # int 0-15 or -1
            "mask_a": torch.from_numpy(mask_a),  # (16,) bool
            "mask_b": torch.from_numpy(mask_b),  # (16,) bool
            "tera_label": tera_label,      # int 0-2
            "opp_action_a": opp_a_slot,    # int 0-15 or -1
            "opp_action_b": opp_b_slot,    # int 0-15 or -1
            "opp_mask_a": torch.from_numpy(opp_mask_a),  # (16,) bool
            "opp_mask_b": torch.from_numpy(opp_mask_b),  # (16,) bool
            "opp_tera_label": opp_tera,    # int 0-2
            "strategic_mask_a": torch.from_numpy(strat_mask_a),      # (16,) bool
            "strategic_mask_b": torch.from_numpy(strat_mask_b),      # (16,) bool
            "opp_strategic_mask_a": torch.from_numpy(opp_strat_mask_a),  # (16,) bool
            "opp_strategic_mask_b": torch.from_numpy(opp_strat_mask_b),  # (16,) bool
            "hp_delta": hp_delta,          # (4,) HP lost
            "ko_flags": ko_flags,          # (4,) binary
            "field_after": field_after,    # (5,)
        }


def build_dataloaders(
    split_dir: str | Path,
    batch_size: int = 512,
    num_workers: int = 4,
    vocab_path: str | Path | None = None,
    require_both_actions: bool = False,
    canonicalize_targets: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader, Vocab]:
    """Build train/val/test DataLoaders."""
    split_dir = Path(split_dir)
    train_path = split_dir / "train.jsonl"
    val_path = split_dir / "val.jsonl"
    test_path = split_dir / "test.jsonl"

    if vocab_path is None:
        print("Building vocab from train split...")
        train_examples = list(read_jsonl(train_path))
        vocab = Vocab.from_examples(train_examples)
        vocab.save(split_dir / "vocab.json")
    else:
        vocab = Vocab.load(vocab_path)
        train_examples = None

    train_ds = Turn1Dataset(train_path, vocab, require_both_actions,
                            canonicalize_targets=canonicalize_targets)
    val_ds = Turn1Dataset(val_path, vocab,
                          canonicalize_targets=canonicalize_targets)
    test_ds = Turn1Dataset(test_path, vocab,
                           canonicalize_targets=canonicalize_targets)

    if train_examples is not None:
        if require_both_actions:
            train_examples = [
                ex for ex in train_examples
                if (ex["action"]["action_a"] is not None
                    and ex["action"]["action_b"] is not None)
            ]
        train_ds.examples = train_examples

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )

    return train_loader, val_loader, test_loader, vocab
