#!/usr/bin/env python3
"""Build a battle_id → split mapping from TurnZero's assembled data.

This allows TurnOne to reuse the same train/val/test splits,
preventing team info leakage across splits.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from turnone.data.io_utils import read_jsonl


def main():
    regime_dir = Path("/home/walter/CS229/turnzero/data/assembled/regime_a")
    out_path = Path("data/split_map.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    battle_to_split: dict[str, str] = {}

    for split_name in ("train", "val", "test"):
        jsonl_path = regime_dir / f"{split_name}.jsonl"
        count = 0
        for ex in read_jsonl(jsonl_path):
            bid = ex["battle_id"]
            if bid not in battle_to_split:
                battle_to_split[bid] = split_name
            count += 1
        print(f"{split_name}: {count} examples, {len([b for b, s in battle_to_split.items() if s == split_name])} unique battles")

    print(f"\nTotal unique battles with split assignment: {len(battle_to_split)}")

    with open(out_path, "w") as f:
        json.dump(battle_to_split, f)

    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
