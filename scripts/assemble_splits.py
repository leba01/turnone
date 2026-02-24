#!/usr/bin/env python3
"""Assemble parsed turn-1 examples into train/val/test splits.

Reuses TurnZero's battle_id → split mapping to prevent team info leakage.
Battles not in the TurnZero split map are assigned to train by default.
"""

import json
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from turnone.data.io_utils import read_jsonl, write_manifest


def main():
    parsed_dirs = [
        Path("data/parsed/gen9vgc2024regg"),
        Path("data/parsed/gen9vgc2024reggbo3"),
        Path("data/parsed/gen9vgc2025reggbo3"),
    ]
    split_map_path = Path("data/split_map.json")
    out_dir = Path("data/assembled")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load split map
    print("Loading split map...")
    with open(split_map_path) as f:
        split_map = json.load(f)
    print(f"  {len(split_map)} battles in split map")

    # Count examples per split
    split_counts = Counter()
    split_files = {}
    for split_name in ("train", "val", "test"):
        split_files[split_name] = open(out_dir / f"{split_name}.jsonl", "w")

    n_unmapped = 0
    n_total = 0
    n_both_actions = 0  # examples where both mons acted (no None actions)

    # Action statistics
    move_counter = Counter()
    target_counter = Counter()
    tera_counter = Counter()

    for parsed_dir in parsed_dirs:
        jsonl_path = parsed_dir / "turn1_examples.jsonl"
        if not jsonl_path.exists():
            print(f"  Skipping {jsonl_path} (not found)")
            continue

        print(f"Processing {jsonl_path}...")
        for ex in read_jsonl(jsonl_path):
            n_total += 1
            battle_id = ex["battle_id"]

            # Look up split from TurnZero mapping
            split_name = split_map.get(battle_id, "train")
            if battle_id not in split_map:
                n_unmapped += 1

            # Add split assignment to example
            ex["split"] = split_name
            split_files[split_name].write(
                json.dumps(ex, separators=(",", ":")) + "\n"
            )
            split_counts[split_name] += 1

            # Stats
            action = ex["action"]
            if action["action_a"] is not None and action["action_b"] is not None:
                n_both_actions += 1
                for key in ("action_a", "action_b"):
                    act = action[key]
                    move_counter[act["move_idx"]] += 1
                    target_counter[act["target"]] += 1

            tera_counter[action["tera_flag"]] += 1

    for f in split_files.values():
        f.close()

    print(f"\nAssembly complete:")
    print(f"  Total examples: {n_total}")
    for split_name in ("train", "val", "test"):
        print(f"  {split_name}: {split_counts[split_name]}")
    print(f"  Unmapped (defaulted to train): {n_unmapped}")
    print(f"  Both mons acted: {n_both_actions}/{n_total} "
          f"({100*n_both_actions/n_total:.1f}%)")

    print(f"\nMove index distribution: {dict(move_counter.most_common())}")
    print(f"Target distribution: {dict(target_counter.most_common())}")
    print(f"Tera distribution: {dict(tera_counter.most_common())}")

    manifest = {
        "total_examples": n_total,
        "split_counts": dict(split_counts),
        "unmapped_battles": n_unmapped,
        "both_actions_count": n_both_actions,
        "both_actions_rate": round(n_both_actions / n_total, 4) if n_total else 0,
        "move_idx_distribution": dict(move_counter),
        "target_distribution": dict(target_counter),
        "tera_distribution": dict(tera_counter),
    }
    write_manifest(out_dir / "assemble_manifest.json", manifest)


if __name__ == "__main__":
    main()
