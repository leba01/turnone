"""Step 2: Join turn-1 reward with battle outcome."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from turnone.data.parser import parse_turn1, SkipBattle
from turnone.rl.reward import compute_reward

RAW_PATH = Path(__file__).resolve().parent.parent / "data/raw/logs-gen9vgc2025reggbo3.json"
OUTCOMES_PATH = Path(__file__).resolve().parent.parent / "results/battle_outcomes/outcomes.jsonl"
OUT_DIR = Path(__file__).resolve().parent.parent / "results/battle_outcomes"

# Field encoding (same as dataset.py)
WEATHER_VOCAB = {
    "none": 0, "RainDance": 1, "SunnyDay": 2, "Sandstorm": 3, "Snow": 4,
    "Snowscape": 4, "DesolateLand": 2,
}
TERRAIN_VOCAB = {
    "none": 0, "Electric Terrain": 1, "Grassy Terrain": 2,
    "Misty Terrain": 3, "Psychic Terrain": 4,
}


def encode_field(field_dict: dict) -> np.ndarray:
    """Encode field state as (5,) float array."""
    weather = field_dict.get("weather") or "none"
    terrain = field_dict.get("terrain") or "none"
    return np.array([
        WEATHER_VOCAB.get(weather, 0),
        TERRAIN_VOCAB.get(terrain, 0),
        float(field_dict.get("trick_room", False)),
        float(field_dict.get("tailwind_p1", False)),
        float(field_dict.get("tailwind_p2", False)),
    ], dtype=np.float64)


def compute_turn1_reward_p1(parsed: dict) -> float:
    """Compute turn-1 reward from P1's perspective using parsed battle data."""
    res = parsed["resolution"]

    # Build arrays from P1's perspective (our = p1, opp = p2)
    hp_delta = np.array([[
        res.hp_before["p1a"] - res.hp_after["p1a"],
        res.hp_before["p1b"] - res.hp_after["p1b"],
        res.hp_before["p2a"] - res.hp_after["p2a"],
        res.hp_before["p2b"] - res.hp_after["p2b"],
    ]], dtype=np.float64)

    ko_flags = np.array([[
        1.0 if "p1a" in res.kos else 0.0,
        1.0 if "p1b" in res.kos else 0.0,
        1.0 if "p2a" in res.kos else 0.0,
        1.0 if "p2b" in res.kos else 0.0,
    ]], dtype=np.float64)

    pre_field = parsed["pre_field"]
    field_before = encode_field(pre_field.to_dict()).reshape(1, -1)
    field_after = encode_field(res.field_state.to_dict()).reshape(1, -1)

    return float(compute_reward(hp_delta, ko_flags, field_before, field_after)[0])


def main():
    # Load outcomes to know which battles have a winner
    print("Loading outcomes...")
    outcomes = {}
    with open(OUTCOMES_PATH) as f:
        for line in f:
            rec = json.loads(line)
            outcomes[rec["battle_id"]] = rec
    print(f"  {len(outcomes)} battles with winner")

    print(f"Loading {RAW_PATH} ...")
    t0 = time.time()
    with open(RAW_PATH) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} battles in {time.time() - t0:.1f}s")

    out_path = OUT_DIR / "turn1_vs_outcome.jsonl"
    n_joined = 0
    n_skip = 0
    n_error = 0

    t0 = time.time()
    with open(out_path, "w") as f:
        for i, (bid, (ts, log_text)) in enumerate(data.items()):
            if bid not in outcomes:
                continue
            outcome = outcomes[bid]

            try:
                parsed = parse_turn1(log_text)
            except (SkipBattle, ValueError, Exception):
                n_skip += 1
                continue

            try:
                reward_p1 = compute_turn1_reward_p1(parsed)
            except Exception:
                n_error += 1
                continue

            # Tera on turn 1?
            from turnone.data.action_space import TERA_NONE
            p1_tera_t1 = parsed["tera_by_side"]["p1"] != TERA_NONE
            p2_tera_t1 = parsed["tera_by_side"]["p2"] != TERA_NONE

            rec = {
                "battle_id": bid,
                "winner": outcome["winner"],
                "turn1_reward_p1": round(reward_p1, 4),
                "p1_tera_turn1": p1_tera_t1,
                "p2_tera_turn1": p2_tera_t1,
                "total_turns": outcome["total_turns"],
            }
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
            n_joined += 1

            if (i + 1) % 50000 == 0:
                print(f"  {i + 1}/{len(data)} processed ...")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Joined: {n_joined}")
    print(f"  Skipped (parse): {n_skip}")
    print(f"  Errors (reward): {n_error}")


if __name__ == "__main__":
    main()
