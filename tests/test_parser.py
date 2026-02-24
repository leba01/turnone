"""Tests for turn-1 parser."""

import json
import pytest
from pathlib import Path

from turnone.data.parser import (
    parse_showteam_line,
    parse_turn1,
    produce_directed_examples,
    camel_to_display,
    SkipBattle,
    _parse_hp,
    _resolve_target,
    _match_move_to_ots,
)


class TestCamelToDisplay:
    def test_basic(self):
        assert camel_to_display("FakeOut") == "Fake Out"

    def test_exception(self):
        assert camel_to_display("Uturn") == "U-turn"
        assert camel_to_display("SwordofRuin") == "Sword of Ruin"

    def test_unk(self):
        assert camel_to_display("UNK") == "UNK"

    def test_idempotent(self):
        assert camel_to_display("Fake Out") == "Fake Out"

    def test_behemoth(self):
        assert camel_to_display("behemothblade") == "Behemoth Blade"


class TestParseHP:
    def test_normal(self):
        assert _parse_hp("34/100") == 34.0

    def test_faint(self):
        assert _parse_hp("0 fnt") == 0.0

    def test_status(self):
        assert _parse_hp("87/100 par") == 87.0

    def test_full(self):
        assert _parse_hp("100/100") == 100.0


class TestResolveTarget:
    def test_self(self):
        assert _resolve_target("p1a", "p1a", "p1") == 3

    def test_ally(self):
        assert _resolve_target("p1a", "p1b", "p1") == 2

    def test_opp_a(self):
        assert _resolve_target("p1a", "p2a", "p1") == 0

    def test_opp_b(self):
        assert _resolve_target("p1a", "p2b", "p1") == 1


class TestMatchMoveToOTS:
    def test_exact(self):
        moves = ["Fake Out", "Flare Blitz", "Protect", "U-turn"]
        assert _match_move_to_ots("Fake Out", moves) == 0
        assert _match_move_to_ots("U-turn", moves) == 3

    def test_fuzzy(self):
        moves = ["Fake Out", "Flare Blitz", "Protect", "U-turn"]
        assert _match_move_to_ots("U-Turn", moves) == 3

    def test_not_found(self):
        moves = ["Fake Out", "Flare Blitz", "Protect", "U-turn"]
        assert _match_move_to_ots("Transform", moves) is None


class TestParseShowteam:
    def test_basic(self):
        line = "|showteam|p1|Rillaboom||AssaultVest|GrassySurge|FakeOut,WoodHammer,GrassyGlide,HighHorsepower|||F|||50|,,,,,Ground"
        # This is a single-mon line; won't have 6 mons
        # Just test it doesn't crash and extracts the first mon
        with pytest.raises(ValueError, match="Expected 6"):
            parse_showteam_line(line)


class TestParseHP2:
    """Integration test for HP parsing edge cases."""

    def test_fractional(self):
        assert abs(_parse_hp("50/100") - 50.0) < 0.01

    def test_non_100_max(self):
        # Some formats use actual HP values
        assert abs(_parse_hp("87/207") - 42.03) < 0.1


# Integration tests require raw data and are slow; mark them
RAW_DATA_PATH = Path("data/raw/logs-gen9vgc2024regg.json")


@pytest.fixture
def sample_battles():
    """Load first 50 battles from raw data."""
    if not RAW_DATA_PATH.exists():
        pytest.skip("Raw data not available")
    with open(RAW_DATA_PATH) as f:
        data = json.load(f)
    return {k: data[k] for k in list(data.keys())[:50]}


class TestParseTurn1Integration:
    def test_parse_some_battles(self, sample_battles):
        """At least some battles should parse successfully."""
        successes = 0
        skips = 0
        errors = 0
        for bid, (ts, log_text) in sample_battles.items():
            try:
                parsed = parse_turn1(log_text)
                examples = produce_directed_examples(bid, parsed, "test")
                assert len(examples) == 2
                for ex in examples:
                    assert ex.battle_id == bid
                    assert len(ex.team_a) == 6
                    assert len(ex.team_b) == 6
                    assert len(ex.lead_indices_a) == 2
                successes += 1
            except SkipBattle:
                skips += 1
            except Exception:
                errors += 1

        assert successes > 20  # at least 40% should parse
        print(f"Parsed {successes}, skipped {skips}, errors {errors}")

    def test_no_crash_on_all(self, sample_battles):
        """Parser should not crash (only raise SkipBattle or ValueError)."""
        for bid, (ts, log_text) in sample_battles.items():
            try:
                parsed = parse_turn1(log_text)
                produce_directed_examples(bid, parsed, "test")
            except (SkipBattle, ValueError):
                pass  # expected
