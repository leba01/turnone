"""Turn-1 parser for VGC doubles battles.

Extracts turn-1 actions (moves, targets, tera), resolution (HP changes, KOs,
field state), and pre-turn ability effects from Showdown protocol logs.

Builds on TurnZero's parser for |showteam| extraction and species matching.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from turnone.data.action_space import (
    Turn1Action,
    JointAction,
    get_target_category,
    move_target_to_slot,
    TERA_NONE,
    TERA_A,
    TERA_B,
)
from turnone.data.io_utils import write_manifest

log = logging.getLogger(__name__)


# ── Name normalization (from TurnZero canonicalize.py) ────────────────────

_NAME_EXCEPTIONS: dict[str, str] = {
    "SwordofRuin": "Sword of Ruin",
    "TabletsofRuin": "Tablets of Ruin",
    "VesselofRuin": "Vessel of Ruin",
    "BeadsofRuin": "Beads of Ruin",
    "PowerofAlchemy": "Power of Alchemy",
    "Uturn": "U-turn",
    "WillOWisp": "Will-O-Wisp",
    "FreezeDry": "Freeze-Dry",
    "XScissor": "X-Scissor",
    "PowerUpPunch": "Power-Up Punch",
    "behemothblade": "Behemoth Blade",
    "behemothbash": "Behemoth Bash",
}

_CAMEL_SPLIT_RE = re.compile(r"(?<=[a-z])(?=[A-Z])")


def camel_to_display(name: str) -> str:
    """Convert CamelCase name to canonical display format."""
    if name == "UNK":
        return name
    if name in _NAME_EXCEPTIONS:
        return _NAME_EXCEPTIONS[name]
    return _CAMEL_SPLIT_RE.sub(" ", name)


# ── Data schemas ──────────────────────────────────────────────────────────

@dataclass
class Pokemon:
    """Single Pokemon with OTS fields."""
    species: str
    item: str = "UNK"
    ability: str = "UNK"
    tera_type: str = "UNK"
    moves: list[str] = field(default_factory=lambda: ["UNK"] * 4)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Pokemon:
        return cls(**d)


@dataclass
class FieldState:
    """Field state at a point in the battle."""
    weather: str | None = None       # "RainDance", "SunnyDay", "Sandstorm", "Snow", None
    terrain: str | None = None       # "Electric Terrain", "Grassy Terrain", etc.
    trick_room: bool = False
    tailwind_p1: bool = False
    tailwind_p2: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FieldState:
        return cls(**d)


@dataclass
class Turn1Resolution:
    """Outcome of turn 1: HP changes, KOs, field state changes."""
    # HP after turn 1 (percentage, 0-100). None if not active.
    hp_after: dict[str, float] = field(default_factory=dict)  # "p1a"->hp, ...
    # HP before turn 1 (always 100 for turn 1, but included for completeness)
    hp_before: dict[str, float] = field(default_factory=dict)
    # KO flags
    kos: list[str] = field(default_factory=list)  # positions that fainted: ["p2a", ...]
    # Field state at end of turn 1
    field_state: FieldState = field(default_factory=FieldState)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hp_after": self.hp_after,
            "hp_before": self.hp_before,
            "kos": self.kos,
            "field_state": self.field_state.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Turn1Resolution:
        return cls(
            hp_after=d["hp_after"],
            hp_before=d["hp_before"],
            kos=d["kos"],
            field_state=FieldState.from_dict(d["field_state"]),
        )


@dataclass
class Turn1Example:
    """One directed turn-1 example (from one player's perspective)."""
    example_id: str
    battle_id: str
    perspective: str           # "p1" or "p2"
    team_a: list[dict]         # our team (6 Pokemon dicts, canonicalized)
    team_b: list[dict]         # opponent team (6 Pokemon dicts, canonicalized)
    lead_indices_a: list[int]  # indices in team_a of our 2 leads [idx_A, idx_B]
    lead_indices_b: list[int]  # indices in team_b of opponent's 2 leads
    pre_turn_field: FieldState # field state after switch-in abilities, before moves
    action: JointAction        # our joint action
    opponent_action: JointAction  # opponent's joint action
    resolution: Turn1Resolution   # turn-1 outcome
    format_id: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "battle_id": self.battle_id,
            "perspective": self.perspective,
            "team_a": self.team_a,
            "team_b": self.team_b,
            "lead_indices_a": self.lead_indices_a,
            "lead_indices_b": self.lead_indices_b,
            "pre_turn_field": self.pre_turn_field.to_dict(),
            "action": self.action.to_dict(),
            "opponent_action": self.opponent_action.to_dict(),
            "resolution": self.resolution.to_dict(),
            "format_id": self.format_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Turn1Example:
        return cls(
            example_id=d["example_id"],
            battle_id=d["battle_id"],
            perspective=d["perspective"],
            team_a=d["team_a"],
            team_b=d["team_b"],
            lead_indices_a=d["lead_indices_a"],
            lead_indices_b=d["lead_indices_b"],
            pre_turn_field=FieldState.from_dict(d["pre_turn_field"]),
            action=JointAction.from_dict(d["action"]),
            opponent_action=JointAction.from_dict(d["opponent_action"]),
            resolution=Turn1Resolution.from_dict(d["resolution"]),
            format_id=d["format_id"],
            metadata=d.get("metadata", {}),
        )


# ── Showteam parsing (from TurnZero) ─────────────────────────────────────

def parse_showteam_line(raw_line: str) -> tuple[str, list[Pokemon]]:
    """Parse a |showteam|pX|... line into (side, list of 6 Pokemon)."""
    prefix = "|showteam|"
    idx = raw_line.find(prefix)
    if idx < 0:
        raise ValueError(f"Not a showteam line: {raw_line[:80]}")

    rest = raw_line[idx + len(prefix):]
    side, mons_str = rest.split("|", 1)

    pokemon: list[Pokemon] = []
    for mon_str in mons_str.split("]"):
        if not mon_str.strip():
            continue
        fields = mon_str.split("|")
        if len(fields) < 5:
            continue

        species = fields[0]
        item = fields[2] if len(fields) > 2 and fields[2] else "UNK"
        ability = fields[3] if len(fields) > 3 and fields[3] else "UNK"

        moves_raw = fields[4].split(",") if len(fields) > 4 and fields[4] else []
        moves = moves_raw[:4]
        while len(moves) < 4:
            moves.append("UNK")

        tera_type = "UNK"
        last_field = fields[-1]
        if last_field:
            tera_parts = last_field.split(",")
            if tera_parts and tera_parts[-1]:
                tera_type = tera_parts[-1]

        pokemon.append(Pokemon(
            species=species, item=item, ability=ability,
            tera_type=tera_type, moves=moves,
        ))

    if len(pokemon) != 6:
        raise ValueError(f"Expected 6 mons from showteam, got {len(pokemon)}")
    return side, pokemon


# ── Helpers ───────────────────────────────────────────────────────────────

def _compute_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _parse_ident(ident: str) -> tuple[str, str, str]:
    """Parse 'p1a: Nickname' -> ('p1', 'a', 'Nickname')."""
    ident = ident.strip()
    side_slot = ident.split(":")[0].strip()
    side = side_slot[:2]   # "p1"
    slot = side_slot[2]    # "a" or "b"
    nickname = ident.split(": ", 1)[1]
    return side, slot, nickname


def _pos_from_ident(ident: str) -> str:
    """Parse 'p1a: Nickname' -> 'p1a'."""
    return ident.strip().split(":")[0].strip()[:3]


def _species_from_details(details: str) -> str:
    """Extract species from details like 'Urshifu-Rapid-Strike, L50, M'."""
    return details.split(",")[0].strip()


def _match_to_showteam(switch_species: str, showteam_species: list[str],
                        used: set[int] | None = None) -> int | None:
    """Match a switch species to its showteam index."""
    skip = used or set()

    # Exact
    for i, st in enumerate(showteam_species):
        if i not in skip and switch_species == st:
            return i
    # Prefix
    for i, st in enumerate(showteam_species):
        if i in skip:
            continue
        if switch_species.startswith(st + "-") or st.startswith(switch_species + "-"):
            return i
    # Base species
    sw_base = switch_species.split("-")[0]
    for i, st in enumerate(showteam_species):
        if i in skip:
            continue
        if st.split("-")[0] == sw_base:
            return i
    return None


def _canonicalize_pokemon(p: Pokemon) -> Pokemon:
    """Normalize names and sort moves."""
    return Pokemon(
        species=camel_to_display(p.species),
        item=camel_to_display(p.item),
        ability=camel_to_display(p.ability),
        tera_type=camel_to_display(p.tera_type),
        moves=sorted(
            [camel_to_display(m) for m in p.moves],
            key=lambda m: (m == "UNK", m),
        ),
    )


def _parse_hp(hp_str: str) -> float:
    """Parse HP string like '34/100' or '0 fnt' → percentage (0-100)."""
    hp_str = hp_str.strip()
    if "fnt" in hp_str:
        return 0.0
    # Handle status conditions: "87/100 par", "50/100 brn"
    hp_part = hp_str.split()[0]
    if "/" in hp_part:
        current, max_hp = hp_part.split("/")
        return float(current) / float(max_hp) * 100.0
    return float(hp_part)


def _resolve_target(attacker_pos: str, target_pos: str,
                    our_side: str) -> int:
    """Map protocol target to our target convention (0-3).

    From our perspective:
      opponent slot A (opp's 'a' slot) → 0
      opponent slot B (opp's 'b' slot) → 1
      ally → 2
      self → 3
    """
    att_side = attacker_pos[:2]
    att_slot = attacker_pos[2]
    tgt_side = target_pos[:2]
    tgt_slot = target_pos[2]

    if attacker_pos == target_pos:
        return 3  # self-target
    if att_side == tgt_side:
        return 2  # ally
    # Opponent: slot a=0, slot b=1
    return 0 if tgt_slot == "a" else 1


def _match_move_to_ots(move_display: str, ots_moves: list[str]) -> int | None:
    """Match a protocol move name to the OTS moveset index.

    Both should be in display format after canonicalization.
    Returns move index (0-3) or None if not found.
    """
    # Direct match
    for i, m in enumerate(ots_moves):
        if m == move_display:
            return i

    # Fuzzy: strip spaces/hyphens and compare lowercase
    move_key = move_display.lower().replace(" ", "").replace("-", "")
    for i, m in enumerate(ots_moves):
        m_key = m.lower().replace(" ", "").replace("-", "")
        if m_key == move_key:
            return i

    return None


# ── Core parser ───────────────────────────────────────────────────────────

class SkipBattle(Exception):
    """Raised to skip a battle (voluntary switch, forfeit, etc.)."""
    pass


def parse_turn1(log_text: str) -> dict[str, Any]:
    """Parse a single battle log for turn-1 data.

    Returns a dict with all turn-1 information, or raises SkipBattle
    if the battle should be excluded.

    Exclusion reasons:
    - Voluntary switch as turn-1 action
    - Forfeit/win before turn 2
    - Missing |showteam| or structural issues
    """
    lines = log_text.split("\n")

    # ── Phase 1: Parse |showteam| ──
    showteam: dict[str, list[Pokemon]] = {}
    for line in lines:
        if "|showteam|" in line:
            side, pokemon = parse_showteam_line(line)
            showteam[side] = pokemon

    for side in ("p1", "p2"):
        if side not in showteam:
            raise ValueError(f"No |showteam| for {side}")

    # Canonicalize pokemon
    canon: dict[str, list[Pokemon]] = {}
    for side in ("p1", "p2"):
        canon[side] = [_canonicalize_pokemon(p) for p in showteam[side]]

    # Species lists for matching
    st_species = {s: [p.species for p in showteam[s]] for s in ("p1", "p2")}
    canon_species = {s: [p.species for p in canon[s]] for s in ("p1", "p2")}

    # ── Phase 2: Find |start|, parse leads + pre-turn state ──
    start_idx = None
    for i, line in enumerate(lines):
        if "|start" in line:
            start_idx = i
            break
    if start_idx is None:
        raise ValueError("No |start| marker")

    # Track nicknames → showteam indices
    nick_to_idx: dict[str, dict[str, int]] = {"p1": {}, "p2": {}}
    # Track lead positions: pos → showteam index
    lead_pos: dict[str, int] = {}  # "p1a" → idx, "p1b" → idx, etc.
    # Track lead nicknames: pos → nickname (for matching moves later)
    lead_nick: dict[str, str] = {}  # "p1a" → "Umbreon", etc.
    leads_per_side: dict[str, list[str]] = {"p1": [], "p2": []}

    # Pre-turn field state (from switch-in abilities)
    pre_field = FieldState()

    turn1_idx = None

    for i, line in enumerate(lines[start_idx + 1:], start=start_idx + 1):
        parts = line.split("|")
        if len(parts) < 2:
            continue
        tag = parts[1].strip()

        if tag == "turn":
            turn_num = parts[2].strip() if len(parts) > 2 else ""
            if turn_num == "1":
                turn1_idx = i
                break

        # Lead switch events (before turn 1)
        if tag in ("switch", "drag") and len(parts) >= 5:
            pos = _pos_from_ident(parts[2])
            side, slot, nick = _parse_ident(parts[2])
            sw_sp = _species_from_details(parts[3])

            idx = _match_to_showteam(sw_sp, st_species[side])
            if idx is not None:
                nick_to_idx[side][nick] = idx
                lead_pos[pos] = idx
                lead_nick[pos] = nick
                if pos not in [f"{side}{s}" for s in leads_per_side[side]]:
                    leads_per_side[side].append(slot)

        # Pre-turn ability effects
        if tag == "-weather" and len(parts) >= 3:
            weather = parts[2].strip()
            if "[from]" in line and "ability" in line:
                pre_field.weather = weather

        if tag == "-fieldstart" and len(parts) >= 3:
            field_text = parts[2].strip()
            if "[from]" in line and "ability" in line:
                if "Grassy Terrain" in field_text:
                    pre_field.terrain = "Grassy Terrain"
                elif "Electric Terrain" in field_text:
                    pre_field.terrain = "Electric Terrain"
                elif "Misty Terrain" in field_text:
                    pre_field.terrain = "Misty Terrain"
                elif "Psychic Terrain" in field_text:
                    pre_field.terrain = "Psychic Terrain"

    if turn1_idx is None:
        raise ValueError("No |turn|1 found")

    # Validate we have 2 leads per side
    for side in ("p1", "p2"):
        if len(leads_per_side[side]) < 2:
            raise ValueError(f"{side} has {len(leads_per_side[side])} leads, expected 2")

    # Map lead positions to canonical indices
    # After canonicalization, mons may be reordered — we need to map
    # raw showteam idx → canonical idx
    # For now, we keep raw indices and don't reorder teams (parser outputs
    # teams in raw showteam order with canonical names)
    lead_idx: dict[str, list[int]] = {}
    for side in ("p1", "p2"):
        a_pos = f"{side}a"
        b_pos = f"{side}b"
        if a_pos not in lead_pos or b_pos not in lead_pos:
            raise ValueError(f"Missing lead positions for {side}")
        lead_idx[side] = [lead_pos[a_pos], lead_pos[b_pos]]

    # ── Phase 3: Parse turn-1 events ──
    # Collect events between |turn|1 and |turn|2
    in_turn1 = True
    moves: list[dict] = []        # {pos, move_name, target_pos, spread}
    tera_events: list[dict] = []  # {pos, tera_type}
    damage_events: list[dict] = []  # {pos, hp}
    faint_events: list[str] = []   # [pos, ...]
    field_changes = FieldState()
    field_changes.weather = pre_field.weather  # carry forward
    field_changes.terrain = pre_field.terrain
    voluntary_switches: list[str] = []
    fainted_in_turn = set()
    move_positions = set()  # positions that used a move

    for line in lines[turn1_idx + 1:]:
        parts = line.split("|")
        if len(parts) < 2:
            continue
        tag = parts[1].strip()

        if tag == "turn":
            break  # reached turn 2

        if tag == "win":
            raise SkipBattle("forfeit/win before turn 2")

        # Tera events
        if tag == "-terastallize" and len(parts) >= 4:
            pos = _pos_from_ident(parts[2])
            tera_type = parts[3].strip()
            tera_events.append({"pos": pos, "tera_type": tera_type})

        # Move events
        if tag == "move" and len(parts) >= 4:
            pos = _pos_from_ident(parts[2])
            side_mv, slot_mv, nick_mv = _parse_ident(parts[2])
            move_name = parts[3].strip()
            target_pos = None
            is_spread = False

            if len(parts) >= 5 and parts[4].strip():
                # Target ident (may have [spread] marker)
                target_str = parts[4].strip()
                if ":" in target_str and not target_str.startswith("["):
                    target_pos = _pos_from_ident(target_str)

            # Check for [spread] marker
            for p in parts[4:]:
                if "[spread]" in p:
                    is_spread = True

            moves.append({
                "pos": pos,
                "side": side_mv,
                "slot": slot_mv,
                "nick": nick_mv,
                "move_name": move_name,
                "target_pos": target_pos,
                "is_spread": is_spread,
            })
            move_positions.add(pos)

        # Switch events (check for voluntary + update nick mapping)
        if tag == "switch" and len(parts) >= 4:
            pos = _pos_from_ident(parts[2])
            side_sw, slot_sw, nick_sw = _parse_ident(parts[2])
            sw_sp = _species_from_details(parts[3])
            is_from_move = "[from]" in line
            is_replacement = pos in fainted_in_turn

            # Update nick_to_idx for any in-turn switches
            if nick_sw not in nick_to_idx[side_sw]:
                idx = _match_to_showteam(sw_sp, st_species[side_sw])
                if idx is not None:
                    nick_to_idx[side_sw][nick_sw] = idx

            if not is_from_move and not is_replacement and pos not in move_positions:
                voluntary_switches.append(pos)

        # Damage events
        if tag == "-damage" and len(parts) >= 4:
            pos = _pos_from_ident(parts[2])
            hp = _parse_hp(parts[3])
            damage_events.append({"pos": pos, "hp": hp})

        # Heal events
        if tag == "-heal" and len(parts) >= 4:
            pos = _pos_from_ident(parts[2])
            hp = _parse_hp(parts[3])
            damage_events.append({"pos": pos, "hp": hp})

        # Faint events
        if tag == "faint" and len(parts) >= 3:
            pos = _pos_from_ident(parts[2])
            faint_events.append(pos)
            fainted_in_turn.add(pos)

        # Field state changes
        if tag == "-weather" and len(parts) >= 3:
            weather = parts[2].strip()
            if "[upkeep]" not in line:
                field_changes.weather = weather

        if tag == "-fieldstart" and len(parts) >= 3:
            field_text = parts[2].strip()
            if "Trick Room" in field_text:
                field_changes.trick_room = True
            elif "Grassy Terrain" in field_text:
                field_changes.terrain = "Grassy Terrain"
            elif "Electric Terrain" in field_text:
                field_changes.terrain = "Electric Terrain"
            elif "Misty Terrain" in field_text:
                field_changes.terrain = "Misty Terrain"
            elif "Psychic Terrain" in field_text:
                field_changes.terrain = "Psychic Terrain"

        if tag == "-sidestart" and len(parts) >= 4:
            side_text = parts[2].strip()
            move_text = parts[3].strip()
            side = side_text.split(":")[0].strip()
            if "Tailwind" in move_text:
                if side == "p1":
                    field_changes.tailwind_p1 = True
                else:
                    field_changes.tailwind_p2 = True

    # ── Phase 4: Check for voluntary switches ──
    if voluntary_switches:
        raise SkipBattle(f"voluntary switch on turn 1: {voluntary_switches}")

    # ── Phase 5: Build actions per side (keyed by nickname for Ally Switch safety) ──
    moves_by_nick: dict[str, dict[str, dict]] = {"p1": {}, "p2": {}}
    for mv in moves:
        side = mv["side"]
        nick = mv["nick"]
        if nick not in moves_by_nick[side]:
            moves_by_nick[side][nick] = mv  # first move per mon

    # Build tera flag per side
    tera_by_side: dict[str, int] = {"p1": TERA_NONE, "p2": TERA_NONE}
    for te in tera_events:
        side = te["pos"][:2]
        slot = te["pos"][2]
        if slot == "a":
            tera_by_side[side] = TERA_A
        else:
            tera_by_side[side] = TERA_B

    # Build resolution
    # HP tracking: start at 100 for all active, then apply last damage event
    hp_before = {}
    hp_after = {}
    for side in ("p1", "p2"):
        for slot in ("a", "b"):
            pos = f"{side}{slot}"
            hp_before[pos] = 100.0
            hp_after[pos] = 100.0

    # Apply damage events in order (last event per position wins)
    for de in damage_events:
        hp_after[de["pos"]] = de["hp"]

    # Override fainted mons
    for pos in faint_events:
        hp_after[pos] = 0.0

    resolution = Turn1Resolution(
        hp_after=hp_after,
        hp_before=hp_before,
        kos=faint_events,
        field_state=field_changes,
    )

    return {
        "showteam": showteam,
        "canon": canon,
        "st_species": st_species,
        "canon_species": canon_species,
        "nick_to_idx": nick_to_idx,
        "lead_pos": lead_pos,
        "lead_nick": lead_nick,
        "lead_idx": lead_idx,
        "pre_field": pre_field,
        "moves_by_nick": moves_by_nick,
        "tera_by_side": tera_by_side,
        "resolution": resolution,
        "moves_raw": moves,
    }


def _build_turn1_action(
    side: str,
    slot: str,
    lead_nick: dict[str, str],
    moves_by_nick: dict[str, dict[str, dict]],
    nick_to_idx: dict[str, dict[str, int]],
    canon: dict[str, list[Pokemon]],
    our_side: str,
) -> Turn1Action | None:
    """Build a Turn1Action for one active mon.

    Returns None if the mon didn't act (fainted before acting).
    Uses the nickname from the |move| event to look up the correct
    mon in the OTS (handles Ally Switch, position changes).
    """
    pos = f"{side}{slot}"
    nick = lead_nick.get(pos)
    if nick is None:
        return None

    if nick not in moves_by_nick[side]:
        return None  # fainted/flinched before acting (no move logged)

    mv = moves_by_nick[side][nick]
    move_name = mv["move_name"]
    target_pos = mv["target_pos"]
    move_pos = mv["pos"]

    # Use nickname to find OTS index (handles Ally Switch correctly)
    idx = nick_to_idx[side].get(nick)
    if idx is None:
        log.debug("Could not find nick '%s' in nick_to_idx for %s", nick, side)
        return None

    mon = canon[side][idx]
    move_idx = _match_move_to_ots(move_name, mon.moves)
    if move_idx is None:
        log.debug("Could not match move '%s' to OTS %s for %s (species=%s)",
                   move_name, mon.moves, pos, mon.species)
        return None

    # Determine target
    cat = get_target_category(move_name)
    if cat in ("self", "spread"):
        target = 3
    elif cat == "ally":
        target = 2
    elif target_pos is not None:
        target = _resolve_target(move_pos, target_pos, our_side)
    else:
        target = 3  # fallback: self/no-target

    return Turn1Action.from_move_target(move_idx, target)


def produce_directed_examples(
    battle_id: str,
    parsed: dict[str, Any],
    format_id: str,
) -> list[Turn1Example]:
    """Produce two directed Turn1Examples (one per player perspective)."""
    examples = []

    for our_side, opp_side, persp in [("p1", "p2", "p1"), ("p2", "p1", "p2")]:
        # Build actions for our side
        action_a = _build_turn1_action(
            our_side, "a", parsed["lead_nick"],
            parsed["moves_by_nick"], parsed["nick_to_idx"],
            parsed["canon"], our_side,
        )
        action_b = _build_turn1_action(
            our_side, "b", parsed["lead_nick"],
            parsed["moves_by_nick"], parsed["nick_to_idx"],
            parsed["canon"], our_side,
        )

        # Build opponent actions
        opp_action_a = _build_turn1_action(
            opp_side, "a", parsed["lead_nick"],
            parsed["moves_by_nick"], parsed["nick_to_idx"],
            parsed["canon"], our_side,
        )
        opp_action_b = _build_turn1_action(
            opp_side, "b", parsed["lead_nick"],
            parsed["moves_by_nick"], parsed["nick_to_idx"],
            parsed["canon"], our_side,
        )

        our_joint = JointAction(
            action_a=action_a,
            action_b=action_b,
            tera_flag=parsed["tera_by_side"][our_side],
        )
        opp_joint = JointAction(
            action_a=opp_action_a,
            action_b=opp_action_b,
            tera_flag=parsed["tera_by_side"][opp_side],
        )

        # Remap resolution to our perspective
        # From our perspective: our_a, our_b, opp_a, opp_b
        res = parsed["resolution"]
        hp_before_ours = {
            "our_a": res.hp_before[f"{our_side}a"],
            "our_b": res.hp_before[f"{our_side}b"],
            "opp_a": res.hp_before[f"{opp_side}a"],
            "opp_b": res.hp_before[f"{opp_side}b"],
        }
        hp_after_ours = {
            "our_a": res.hp_after[f"{our_side}a"],
            "our_b": res.hp_after[f"{our_side}b"],
            "opp_a": res.hp_after[f"{opp_side}a"],
            "opp_b": res.hp_after[f"{opp_side}b"],
        }
        kos_ours = []
        for k in res.kos:
            side = k[:2]
            slot = k[2]
            if side == our_side:
                kos_ours.append(f"our_{slot}")
            else:
                kos_ours.append(f"opp_{slot}")

        # Remap field state tailwinds
        tw_ours = (res.field_state.tailwind_p1 if our_side == "p1"
                   else res.field_state.tailwind_p2)
        tw_opp = (res.field_state.tailwind_p2 if our_side == "p1"
                  else res.field_state.tailwind_p1)

        directed_field = FieldState(
            weather=res.field_state.weather,
            terrain=res.field_state.terrain,
            trick_room=res.field_state.trick_room,
            tailwind_p1=tw_ours,   # p1 = our side in directed
            tailwind_p2=tw_opp,    # p2 = opp side in directed
        )

        directed_resolution = Turn1Resolution(
            hp_before=hp_before_ours,
            hp_after=hp_after_ours,
            kos=kos_ours,
            field_state=directed_field,
        )

        # Pre-turn field (also remap tailwinds)
        pf = parsed["pre_field"]
        tw_pre_ours = pf.tailwind_p1 if our_side == "p1" else pf.tailwind_p2
        tw_pre_opp = pf.tailwind_p2 if our_side == "p1" else pf.tailwind_p1
        directed_pre_field = FieldState(
            weather=pf.weather,
            terrain=pf.terrain,
            trick_room=pf.trick_room,
            tailwind_p1=tw_pre_ours,
            tailwind_p2=tw_pre_opp,
        )

        # Team dicts (canonicalized, in showteam order)
        team_a_dicts = [p.to_dict() for p in parsed["canon"][our_side]]
        team_b_dicts = [p.to_dict() for p in parsed["canon"][opp_side]]

        examples.append(Turn1Example(
            example_id=_compute_hash(f"{battle_id}|turn1|{persp}"),
            battle_id=battle_id,
            perspective=persp,
            team_a=team_a_dicts,
            team_b=team_b_dicts,
            lead_indices_a=parsed["lead_idx"][our_side],
            lead_indices_b=parsed["lead_idx"][opp_side],
            pre_turn_field=directed_pre_field,
            action=our_joint,
            opponent_action=opp_joint,
            resolution=directed_resolution,
            format_id=format_id,
            metadata={},
        ))

    return examples


# ── Full pipeline runner ──────────────────────────────────────────────────

def run_parse(raw_path: str, out_dir: str, limit: int | None = None) -> dict[str, Any]:
    """Parse all battles from a raw JSON file for turn-1 data.

    Writes JSONL + manifest. Skips battles with voluntary switches,
    forfeits, or structural issues.
    """
    raw_path = Path(raw_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = raw_path.stem
    format_id = stem.removeprefix("logs-")

    print(f"Loading {raw_path} ...")
    t0 = time.time()
    with open(raw_path) as f:
        data = json.load(f)
    load_time = time.time() - t0
    print(f"Loaded {len(data)} battles in {load_time:.1f}s")

    battle_ids = list(data.keys())
    if limit is not None:
        battle_ids = battle_ids[:limit]

    jsonl_path = out_dir / "turn1_examples.jsonl"

    examples_written = 0
    n_success = 0
    n_skipped = 0
    n_errors = 0
    skip_reasons: dict[str, int] = {}
    error_samples: list[dict[str, str]] = []
    n_missing_action = 0

    t0 = time.time()
    with open(jsonl_path, "w") as f:
        for i, bid in enumerate(battle_ids):
            _ts, log_text = data[bid]
            try:
                parsed = parse_turn1(log_text)
                examples = produce_directed_examples(bid, parsed, format_id)
                for ex in examples:
                    # Track examples where a mon's action is None
                    if ex.action.action_a is None or ex.action.action_b is None:
                        n_missing_action += 1
                    f.write(json.dumps(ex.to_dict(), separators=(",", ":")) + "\n")
                    examples_written += 1
                n_success += 1

            except SkipBattle as e:
                n_skipped += 1
                reason = str(e).split(":")[0] if ":" in str(e) else str(e)
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

            except Exception as e:
                n_errors += 1
                if len(error_samples) < 20:
                    error_samples.append({"battle_id": bid, "error": str(e)})

            if (i + 1) % 10000 == 0:
                print(f"  {i + 1}/{len(battle_ids)} battles processed ...")

    parse_time = time.time() - t0
    print(f"\nParsed {n_success}/{len(battle_ids)} battles in {parse_time:.1f}s")
    print(f"  Skipped: {n_skipped} ({skip_reasons})")
    print(f"  Errors: {n_errors}")
    print(f"  Examples written: {examples_written}")
    print(f"  Examples with missing action (fainted before acting): {n_missing_action}")

    manifest = {
        "raw_path": str(raw_path),
        "format_id": format_id,
        "total_battles": len(data),
        "battles_attempted": len(battle_ids),
        "battles_parsed": n_success,
        "battles_skipped": n_skipped,
        "skip_reasons": skip_reasons,
        "parse_errors": n_errors,
        "examples_written": examples_written,
        "examples_missing_action": n_missing_action,
        "parse_time_seconds": round(parse_time, 1),
        "error_samples": error_samples,
    }
    write_manifest(out_dir / "parse_manifest.json", manifest)
    return manifest
