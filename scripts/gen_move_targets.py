#!/usr/bin/env python3
"""Fetch Pokemon Showdown's moves.json and produce move_targets.json.

Cross-references Showdown's authoritative targeting data with our vocab,
storing raw Showdown target values keyed by our vocab move names.

Usage:
    PYTHONPATH=. python scripts/gen_move_targets.py \
        --vocab_path runs/bc_001/vocab.json \
        --out_path turnone/data/move_targets.json
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path

SHOWDOWN_MOVES_URL = "https://play.pokemonshowdown.com/data/moves.json"


def normalize(name: str) -> str:
    """Normalize a move name for matching: lowercase, strip non-alnum."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def fetch_showdown_moves() -> dict[str, dict]:
    """Fetch and parse Showdown's moves.json."""
    print(f"Fetching {SHOWDOWN_MOVES_URL} ...")
    req = urllib.request.Request(
        SHOWDOWN_MOVES_URL,
        headers={"User-Agent": "TurnOne-CS234/1.0"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read().decode("utf-8")
    data = json.loads(raw)
    print(f"  Fetched {len(data)} Showdown move entries")
    return data


def build_showdown_lookup(showdown_moves: dict[str, dict]) -> dict[str, str]:
    """Build normalized_name → Showdown target value lookup."""
    lookup: dict[str, str] = {}
    for key, entry in showdown_moves.items():
        if not isinstance(entry, dict):
            continue
        target = entry.get("target")
        if target is None:
            continue
        norm = normalize(entry.get("name", key))
        lookup[norm] = target
    return lookup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab_path", type=Path, default=Path("runs/bc_001/vocab.json"))
    parser.add_argument("--out_path", type=Path, default=Path("turnone/data/move_targets.json"))
    args = parser.parse_args()

    # Load our vocab
    with open(args.vocab_path) as f:
        vocab = json.load(f)
    move_vocab: dict[str, int] = vocab["move"]
    move_names = [name for name in move_vocab if name != "<UNK>"]
    print(f"Vocab has {len(move_names)} moves (excluding <UNK>)")

    # Fetch Showdown data
    showdown_moves = fetch_showdown_moves()
    showdown_lookup = build_showdown_lookup(showdown_moves)

    # Match vocab moves to Showdown targets
    move_targets: dict[str, str] = {}
    matched = 0
    unmatched: list[str] = []

    for name in move_names:
        norm = normalize(name)
        target = showdown_lookup.get(norm)
        if target is not None:
            move_targets[name] = target
            matched += 1
        else:
            unmatched.append(name)

    # Coverage report
    total = len(move_names)
    print(f"\nCoverage: {matched}/{total} ({100*matched/total:.1f}%)")
    if unmatched:
        print(f"\nUnmatched moves ({len(unmatched)}):")
        for name in sorted(unmatched):
            print(f"  {name}")

    # Spot-check
    spot_checks = {
        "Protect": "self",
        "Earthquake": "allAdjacent",
        "Helping Hand": "adjacentAlly",
        "Thunderbolt": "normal",
        "Tailwind": "allySide",
        "Trick Room": "all",
    }
    print("\nSpot checks:")
    for move, expected in spot_checks.items():
        actual = move_targets.get(move, "MISSING")
        status = "OK" if actual == expected else f"MISMATCH (expected {expected})"
        print(f"  {move}: {actual} {status}")

    # Write output
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_path, "w") as f:
        json.dump(move_targets, f, indent=2, sort_keys=True)
    print(f"\nWrote {len(move_targets)} entries to {args.out_path}")


if __name__ == "__main__":
    main()
