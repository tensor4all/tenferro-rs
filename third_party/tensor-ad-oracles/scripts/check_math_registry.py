"""Validate the math-note registry against the repository tree."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from validators.math_registry import validate_registry


def main() -> int:
    try:
        validate_registry(REPO_ROOT)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print("math_registry_ok=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
