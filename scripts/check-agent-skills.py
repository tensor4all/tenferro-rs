#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import sys

CANONICAL_SKILL = pathlib.Path(".agents/skills/tenferro-compute")
MIRROR_SKILLS = (
    pathlib.Path(".claude/skills/tenferro-compute"),
    pathlib.Path(".kimi/skills/tenferro-compute"),
)
REFERENCE_FILES = (
    "references/crate-selection.md",
    "references/api-cheatsheet.md",
    "references/performance-idioms.md",
    "references/pitfalls.md",
)
PORTABLE_FILES = ("SKILL.md", *REFERENCE_FILES)
OPENCODE_ENTRY = pathlib.Path(".opencode/commands/tenferro-compute.md")


def check(root: pathlib.Path) -> list[str]:
    errors: list[str] = []
    canonical = root / CANONICAL_SKILL

    for relative in ("SKILL.md", "agents/openai.yaml", *REFERENCE_FILES):
        path = canonical / relative
        if not path.is_file():
            errors.append(f"missing canonical skill file: {CANONICAL_SKILL / relative}")

    if not canonical.is_dir():
        return errors + check_skill_mirrors(root)

    for mirror_relative in MIRROR_SKILLS:
        mirror = root / mirror_relative
        for relative in PORTABLE_FILES:
            canonical_file = canonical / relative
            mirror_file = mirror / relative
            if not mirror_file.is_file():
                errors.append(f"missing mirror file: {mirror_relative / relative}")
            elif not canonical_file.is_file():
                continue
            elif canonical_file.read_bytes() != mirror_file.read_bytes():
                errors.append(f"mirror file does not match canonical: {mirror_relative / relative}")
        if mirror.is_dir():
            actual = {
                path.relative_to(mirror).as_posix()
                for path in mirror.rglob("*.md")
                if path.is_file()
            }
            unexpected = sorted(actual.difference(PORTABLE_FILES))
            for relative in unexpected:
                errors.append(f"unexpected mirror Markdown file: {mirror_relative / relative}")

    errors.extend(check_skill_mirrors(root))

    entry = root / OPENCODE_ENTRY
    if not entry.is_file():
        errors.append(f"missing OpenCode entry: {OPENCODE_ENTRY}")
    else:
        entry_text = entry.read_text(encoding="utf-8")
        for relative in REFERENCE_FILES:
            reference = (CANONICAL_SKILL / relative).as_posix()
            if reference not in entry_text:
                errors.append(f"OpenCode entry is missing reference: {reference}")

    return errors


def check_skill_mirrors(root: pathlib.Path) -> list[str]:
    """Every skill under `.agents/skills` must be mirrored byte-identically.

    `tenferro-compute` has the stricter pass above (canonical-only files plus
    reference links in its OpenCode entry), so it is skipped here to avoid
    reporting the same drift twice.
    """
    errors: list[str] = []
    canonical_root = root / ".agents" / "skills"
    if not canonical_root.is_dir():
        return errors
    for skill in sorted(path for path in canonical_root.iterdir() if path.is_dir()):
        if skill.name == CANONICAL_SKILL.name or not (skill / "SKILL.md").is_file():
            continue
        portable = sorted(
            path.relative_to(skill).as_posix()
            for path in skill.rglob("*.md")
            if path.is_file()
        )
        for mirror_root in MIRROR_SKILLS:
            mirror = root / mirror_root.parent / skill.name
            for relative in portable:
                canonical_file = skill / relative
                mirror_file = mirror / relative
                label = f"{mirror_root.parent.as_posix()}/{skill.name}/{relative}"
                if not mirror_file.is_file():
                    errors.append(f"missing mirror file: {label}")
                elif canonical_file.read_bytes() != mirror_file.read_bytes():
                    errors.append(f"mirror file does not match canonical: {label}")
            if mirror.is_dir():
                actual = {
                    path.relative_to(mirror).as_posix()
                    for path in mirror.rglob("*.md")
                    if path.is_file()
                }
                for relative in sorted(actual.difference(portable)):
                    errors.append(
                        f"unexpected mirror Markdown file: "
                        f"{mirror_root.parent.as_posix()}/{skill.name}/{relative}"
                    )
        entry = root / ".opencode" / "commands" / f"{skill.name}.md"
        if not entry.is_file():
            errors.append(f"missing OpenCode entry: .opencode/commands/{skill.name}.md")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Check bundled tenferro agent-skill mirrors.")
    parser.add_argument("--root-dir", default=pathlib.Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = pathlib.Path(args.root_dir).resolve()
    errors = check(root)
    if errors:
        print("agent-skill layout errors:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("agent-skills-ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
