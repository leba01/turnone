"""Step 1: Extract battle outcomes (winner, tera timing) from raw battle logs."""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

RAW_PATH = Path(__file__).resolve().parent.parent / "data/raw/logs-gen9vgc2025reggbo3.json"
OUT_DIR = Path(__file__).resolve().parent.parent / "results/battle_outcomes"


def parse_battle_outcome(battle_id: str, log_text: str) -> dict | None:
    """Parse a single battle log for outcome + tera timing."""
    lines = log_text.split("\n")

    # Map player names to p1/p2
    p1_name = p2_name = None
    winner_name = None
    winner = None
    total_turns = 0
    p1_tera_turn = None
    p2_tera_turn = None
    current_turn = 1  # actions before |turn|2 = turn 1

    for line in lines:
        parts = line.split("|")
        if len(parts) < 2:
            continue
        tag = parts[1].strip()

        if tag == "player" and len(parts) >= 4:
            side = parts[2].strip()
            name = parts[3].strip()
            if side == "p1":
                p1_name = name
            elif side == "p2":
                p2_name = name

        elif tag == "turn" and len(parts) >= 3:
            turn_num = int(parts[2].strip())
            current_turn = turn_num
            total_turns = turn_num

        elif tag == "-terastallize" and len(parts) >= 4:
            pos = parts[2].strip().split(":")[0].strip()
            side = pos[:2]
            if side == "p1" and p1_tera_turn is None:
                p1_tera_turn = current_turn
            elif side == "p2" and p2_tera_turn is None:
                p2_tera_turn = current_turn

        elif tag == "win" and len(parts) >= 3:
            winner_name = parts[2].strip()

    if winner_name is None:
        return None  # no winner → skip

    if winner_name == p1_name:
        winner = "p1"
    elif winner_name == p2_name:
        winner = "p2"
    else:
        return None  # can't resolve winner

    # total_turns: if we never saw a |turn| tag, it's a very short game
    if total_turns == 0:
        total_turns = 1

    return {
        "battle_id": battle_id,
        "winner": winner,
        "p1_tera_turn": p1_tera_turn,
        "p2_tera_turn": p2_tera_turn,
        "total_turns": total_turns,
        "p1_name": p1_name or "",
        "p2_name": p2_name or "",
    }


def main():
    print(f"Loading {RAW_PATH} ...")
    t0 = time.time()
    with open(RAW_PATH) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} battles in {time.time() - t0:.1f}s")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "outcomes.jsonl"

    n_total = 0
    n_with_winner = 0
    n_no_winner = 0
    n_tera_p1 = 0
    n_tera_p2 = 0

    t0 = time.time()
    with open(out_path, "w") as f:
        for i, (bid, (ts, log_text)) in enumerate(data.items()):
            n_total += 1
            result = parse_battle_outcome(bid, log_text)
            if result is None:
                n_no_winner += 1
                continue
            n_with_winner += 1
            if result["p1_tera_turn"] is not None:
                n_tera_p1 += 1
            if result["p2_tera_turn"] is not None:
                n_tera_p2 += 1
            f.write(json.dumps(result, separators=(",", ":")) + "\n")

            if (i + 1) % 50000 == 0:
                print(f"  {i + 1}/{len(data)} processed ...")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Total battles: {n_total}")
    print(f"  With winner: {n_with_winner}")
    print(f"  No winner (skipped): {n_no_winner}")
    print(f"  P1 tera'd at some point: {n_tera_p1} ({n_tera_p1/n_with_winner*100:.1f}%)")
    print(f"  P2 tera'd at some point: {n_tera_p2} ({n_tera_p2/n_with_winner*100:.1f}%)")


if __name__ == "__main__":
    main()
