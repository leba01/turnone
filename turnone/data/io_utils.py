"""JSONL and manifest I/O helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


def write_jsonl(path: str | Path, records: Iterator[dict[str, Any]] | list[dict[str, Any]]) -> int:
    """Write records as newline-delimited JSON. Returns count written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield dicts from a JSONL file."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    """Write a JSON manifest file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
