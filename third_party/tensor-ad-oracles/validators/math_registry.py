"""Validation helpers for the math-note registry."""

from __future__ import annotations

import json
import re
from pathlib import Path


REGISTRY_PATH = Path("docs/math/registry.json")
_HTML_ID_RE = re.compile(r"<a\s+id=\"([A-Za-z0-9._-]+)\"\s*></a>")
_ATTRIBUTE_ID_RE = re.compile(r"\{#([A-Za-z0-9._-]+)\}")


def load_registry(root: Path) -> dict:
    """Load the math-note registry from the repository root."""
    path = root / REGISTRY_PATH
    if not path.exists():
        raise ValueError(f"registry not found: {REGISTRY_PATH}")
    return json.loads(path.read_text(encoding="utf-8"))


def materialized_case_families(cases_root: Path) -> set[tuple[str, str]]:
    """Return every materialized `(op, family)` pair under `cases/`."""
    families: set[tuple[str, str]] = set()
    if not cases_root.exists():
        return families
    for path in sorted(cases_root.rglob("*.jsonl")):
        relative = path.relative_to(cases_root)
        if len(relative.parts) != 2:
            continue
        families.add((relative.parts[0], path.stem))
    return families


def extract_markdown_anchors(text: str) -> set[str]:
    """Extract explicit anchor IDs from a markdown document."""
    anchors = set(_HTML_ID_RE.findall(text))
    anchors.update(_ATTRIBUTE_ID_RE.findall(text))
    return anchors


def _resolve_note_path(root: Path, note_path: str) -> Path:
    candidate = Path(note_path)
    if candidate.is_absolute():
        raise ValueError(f"note_path must be repo-relative: {note_path}")
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"note_path escapes repository root: {note_path}")
    return resolved


def validate_registry(root: Path) -> None:
    """Validate registry structure and coverage against the case tree."""
    root = root.resolve()
    registry = load_registry(root)
    entries = registry.get("entries")
    if not isinstance(entries, list):
        raise ValueError("registry entries must be a list")

    seen: set[tuple[str, str]] = set()
    covered: set[tuple[str, str]] = set()
    note_anchor_cache: dict[Path, set[str]] = {}

    for entry in entries:
        op = entry.get("op")
        family = entry.get("family")
        note_path = entry.get("note_path")
        anchor = entry.get("anchor")

        if not all(isinstance(value, str) and value for value in (op, family, note_path, anchor)):
            raise ValueError(f"invalid registry entry: {entry!r}")

        key = (op, family)
        if key in seen:
            raise ValueError(f"duplicate registry entry for {op}/{family}")
        seen.add(key)

        resolved_note_path = _resolve_note_path(root, note_path)
        if not resolved_note_path.exists():
            raise ValueError(f"note_path not found: {note_path}")

        if resolved_note_path not in note_anchor_cache:
            note_anchor_cache[resolved_note_path] = extract_markdown_anchors(
                resolved_note_path.read_text(encoding="utf-8")
            )
        if anchor not in note_anchor_cache[resolved_note_path]:
            raise ValueError(f"missing anchor {anchor} in {note_path}")

        covered.add(key)

    expected = materialized_case_families(root / "cases")
    missing = sorted(expected - covered)
    if missing:
        formatted = ", ".join(f"{op}/{family}" for op, family in missing)
        raise ValueError(f"missing registry entries for materialized case families: {formatted}")
