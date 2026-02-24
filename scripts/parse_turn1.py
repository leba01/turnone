#!/usr/bin/env python3
"""Parse raw battle logs for turn-1 data."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from turnone.data.parser import run_parse


def main():
    parser = argparse.ArgumentParser(description="Parse turn-1 data from battle logs")
    parser.add_argument("raw_path", help="Path to raw JSON battle log file")
    parser.add_argument("--out-dir", default="data/parsed", help="Output directory")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of battles")
    args = parser.parse_args()

    manifest = run_parse(args.raw_path, args.out_dir, limit=args.limit)
    print(f"\nDone. Manifest: {manifest}")


if __name__ == "__main__":
    main()
