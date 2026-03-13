"""Tests for action space encoding."""

import json
from pathlib import Path

import numpy as np
import pytest

from turnone.data.action_space import (
    move_target_to_slot,
    slot_to_move_target,
    get_target_category,
    compute_action_mask,
    Turn1Action,
    JointAction,
    SLOTS_PER_MON,
    NUM_TERA,
    TERA_NONE,
    TERA_A,
    TERA_B,
    _MOVE_TARGETS,
)


class TestSlotEncoding:
    def test_roundtrip(self):
        for move_idx in range(4):
            for target in range(4):
                slot = move_target_to_slot(move_idx, target)
                m, t = slot_to_move_target(slot)
                assert m == move_idx
                assert t == target

    def test_slot_range(self):
        slots = set()
        for m in range(4):
            for t in range(4):
                s = move_target_to_slot(m, t)
                assert 0 <= s < SLOTS_PER_MON
                slots.add(s)
        assert len(slots) == SLOTS_PER_MON


class TestTargetCategory:
    @pytest.mark.parametrize("move,expected", [
        ("Protect", "self"),
        ("Earthquake", "spread"),
        ("Helping Hand", "ally"),
        ("Fake Out", "single"),
        ("UNK", "single"),
        ("Trick Room", "self"),
        ("Tailwind", "self"),
        ("Meteor Beam", "single"),  # Showdown "normal" — player chooses target
        # Previously-misclassified moves (caught by audit)
        ("Outrage", "self"),
        ("Counter", "self"),
        ("Air Cutter", "spread"),
        ("Dragon Cheer", "ally"),
        ("Scary Face", "single"),
        ("Self Destruct", "spread"),
        ("Soft Boiled", "self"),
        ("Acupressure", "self"),
    ])
    def test_target_category(self, move, expected):
        assert get_target_category(move) == expected


class TestMoveTargetsJson:
    """Verify move_targets.json is loaded and has full vocab coverage."""

    def test_vocab_coverage(self):
        vocab_path = Path("runs/bc_001/vocab.json")
        if not vocab_path.exists():
            pytest.skip("vocab.json not available")
        with open(vocab_path) as f:
            vocab = json.load(f)
        move_names = [n for n in vocab["move"] if n != "<UNK>"]
        missing = [n for n in move_names if n not in _MOVE_TARGETS]
        assert not missing, f"Moves missing from move_targets.json: {missing}"


class TestActionMask:
    def test_basic_moveset(self):
        """Standard moveset with mix of types."""
        moves = ["Fake Out", "Flare Blitz", "Protect", "Earthquake"]
        mask = compute_action_mask(moves)
        assert mask.shape == (16,)
        assert mask.dtype == bool

        # Fake Out (single): targets 0,1,2,3 all valid
        assert mask[move_target_to_slot(0, 0)]  # opp A
        assert mask[move_target_to_slot(0, 1)]  # opp B
        assert mask[move_target_to_slot(0, 2)]  # ally
        assert mask[move_target_to_slot(0, 3)]  # self

        # Protect (self): only target 3
        assert not mask[move_target_to_slot(2, 0)]
        assert not mask[move_target_to_slot(2, 1)]
        assert mask[move_target_to_slot(2, 3)]

        # Earthquake (spread): only target 3
        assert not mask[move_target_to_slot(3, 0)]
        assert mask[move_target_to_slot(3, 3)]

    def test_unk_moves_masked(self):
        moves = ["UNK", "UNK", "UNK", "UNK"]
        mask = compute_action_mask(moves)
        assert not mask.any()

class TestTurn1Action:
    def test_roundtrip_dict(self):
        act = Turn1Action(move_idx=1, target=2, slot=6)
        d = act.to_dict()
        act2 = Turn1Action.from_dict(d)
        assert act == act2


class TestJointAction:
    def test_with_none(self):
        ja = JointAction(
            action_a=Turn1Action(0, 0, 0),
            action_b=None,
            tera_flag=TERA_A,
        )
        d = ja.to_dict()
        assert d["action_b"] is None
        ja2 = JointAction.from_dict(d)
        assert ja2.action_b is None
        assert ja2.tera_flag == TERA_A
