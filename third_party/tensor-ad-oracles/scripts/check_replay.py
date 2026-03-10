"""Replay the published JSON database and require zero failures."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CASES_ROOT = REPO_ROOT / "cases"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from validators.replay import replay_case_tree


def main() -> int:
    result = replay_case_tree(CASES_ROOT)
    if result.failures:
        joined = "\n".join(result.failures)
        raise SystemExit(f"replay failed:\n{joined}")
    print(f"replay_checked={result.checked}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
