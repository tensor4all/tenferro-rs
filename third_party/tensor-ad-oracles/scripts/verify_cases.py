"""Helpers for checking case-file hygiene."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
CASES_ROOT = REPO_ROOT / "cases"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from validators.case_loader import iter_case_files


def load_jsonl_records(path: Path) -> list[dict]:
    """Load all JSON objects from a JSONL file."""
    records: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def find_duplicate_case_ids(paths: Iterable[Path]) -> None:
    """Raise when the same case id appears in more than one record."""
    seen: dict[str, Path] = {}
    for path in paths:
        for record in load_jsonl_records(path):
            case_id = record["case_id"]
            previous = seen.get(case_id)
            if previous is not None:
                raise ValueError(
                    f"duplicate case_id {case_id!r} found in {previous} and {path}"
                )
            seen[case_id] = path


def verify_case_tree(root: Path = CASES_ROOT) -> int:
    """Check repository-level case-file hygiene and return the record count."""
    paths = list(iter_case_files(root))
    find_duplicate_case_ids(paths)
    return sum(len(load_jsonl_records(path)) for path in paths)


def main() -> int:
    verified = verify_case_tree()
    print(f"verified_case_records={verified}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
