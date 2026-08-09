#!/usr/bin/env python3
"""Regression tests for guide dependency and setup-policy checks."""
from __future__ import annotations

import importlib.util
import pathlib
import sys
import tempfile


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "check_guide_dependency_snippets",
    ROOT / "scripts" / "check-guide-dependency-snippets.py",
)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise RuntimeError("unable to load guide checker")
CHECKER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CHECKER
SPEC.loader.exec_module(CHECKER)


def test_all_guides_reject_commit_checkout_hashes() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        guide = root / "docs" / "guides" / "pinned.md"
        guide.parent.mkdir(parents=True)
        guide.write_text(
            "```bash\ngit clone https://example.invalid/repo.git\n"
            "git checkout 0123456789abcdef\n```\n",
            encoding="utf-8",
        )

        findings = CHECKER.guide_commit_checkout_hashes(root)
        assert findings == [(guide, 3, "0123456789abcdef")]
        try:
            CHECKER.validate_no_guide_commit_checkout_hashes(root)
        except RuntimeError as error:
            assert "pinned.md:3" in str(error)
        else:  # pragma: no cover
            raise AssertionError("commit checkout hash should be rejected")


def test_non_commit_checkout_targets_remain_allowed() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        guide = root / "docs" / "guides" / "branch.md"
        guide.parent.mkdir(parents=True)
        guide.write_text("git checkout main\n", encoding="utf-8")
        CHECKER.validate_no_guide_commit_checkout_hashes(root)
        assert CHECKER.guide_commit_checkout_hashes(root) == []


if __name__ == "__main__":
    test_all_guides_reject_commit_checkout_hashes()
    test_non_commit_checkout_targets_remain_allowed()
