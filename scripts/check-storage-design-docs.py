#!/usr/bin/env python3
"""Validate the normative storage-ownership contract document.

The ledger records the document as an immutable P1 artifact.  This checker is
intentionally small, but it validates the parts of the document that make it
the normative Phase 1 contract rather than accepting any non-empty Markdown
file.  It does not rewrite the document or infer implementation status from
source code.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


EXPECTED_DOCUMENT = Path("docs/design/storage-ownership-contracts.md")
REQUIRED_MARKERS = (
    "# Storage Ownership Contracts",
    "## Phase 1 verification ledger",
    "tenferro.storage-ownership-contracts.v2",
    "scripts/storage-ownership-contracts.toml",
    "trusted-runner execution log",
    "not a security attestation",
    "No content checksum",
    "p0-control-plane",
    "p1-element-access-baseline",
    "p2-root-claims",
    "current production state deliberately activates only",
    "state = { kind = \"deferred\"",
    "tenferro.storage-ownership-receipt.v1",
    "candidate_commit",
    "base_commit",
    "git rev-parse --verify",
    "exit_code",
    "artifact_path",
    "tracked",
    "path_args",
    "atomic",
    "CheckedLayout",
    "contiguous",
)
REQUIRED_GATE_HEADINGS = tuple(f"## G{number}." for number in range(1, 8))


class DocumentCheckError(ValueError):
    """A structured validation failure for the normative document."""


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def confined_document(root: Path, requested: str | None) -> Path:
    relative = EXPECTED_DOCUMENT if requested is None else Path(requested)
    if relative.is_absolute() or any(part == ".." for part in relative.parts):
        raise DocumentCheckError("document path must be repository-relative")
    document = (root / relative).resolve()
    try:
        document.relative_to(root)
    except ValueError as error:
        raise DocumentCheckError("document path escapes the repository") from error
    if document.relative_to(root) != EXPECTED_DOCUMENT:
        raise DocumentCheckError(
            f"document must be {EXPECTED_DOCUMENT.as_posix()}"
        )
    if not document.is_file():
        raise DocumentCheckError(f"document is missing: {EXPECTED_DOCUMENT.as_posix()}")
    return document


def validate(document: Path) -> None:
    try:
        text = document.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise DocumentCheckError(f"cannot read document: {error}") from error
    if not text.strip():
        raise DocumentCheckError("document is empty")
    missing = [marker for marker in REQUIRED_MARKERS if marker not in text]
    missing += [heading for heading in REQUIRED_GATE_HEADINGS if heading not in text]
    if missing:
        raise DocumentCheckError(
            "document is missing required contract markers: " + ", ".join(missing)
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "document",
        nargs="?",
        help=f"repository-relative document (default: {EXPECTED_DOCUMENT.as_posix()})",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=repository_root(),
        help="repository root (used by focused checks)",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    try:
        validate(confined_document(root, args.document))
    except DocumentCheckError as error:
        print(f"storage-design-docs: {error}", file=sys.stderr)
        return 1
    print("storage-design-docs-ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
