"""Load case files from the published JSONL database."""

from __future__ import annotations

import json
from pathlib import Path


def load_case_file(path: Path) -> list[dict]:
    """Load one JSONL case file."""
    records: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def iter_case_files(root: Path) -> list[Path]:
    """List all JSONL case files under a root directory."""
    return sorted(root.rglob("*.jsonl"))
