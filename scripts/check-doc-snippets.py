#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import re
import sys

START_RE = re.compile(r"<!--\s*snippet-source:\s*([^>]+?)\s*-->")
END_RE = re.compile(r"<!--\s*end-snippet-source\s*-->")
REGION_RE = re.compile(r"^\s*//\s*snippet-(start|end):([A-Za-z0-9_-]+)\s*$")


def source_regions(source: pathlib.Path) -> dict[str, str]:
    lines = source.read_text(encoding="utf-8").splitlines(keepends=True)
    regions: dict[str, str] = {}
    active: tuple[str, int] | None = None
    for index, line in enumerate(lines):
        marker = REGION_RE.match(line.rstrip("\r\n"))
        if marker is None:
            continue
        kind, name = marker.groups()
        if kind == "start":
            if active is not None:
                raise ValueError(f"{source}: nested snippet region {name!r}")
            if name in regions:
                raise ValueError(f"{source}: duplicate snippet region {name!r}")
            active = (name, index)
        elif active is None:
            raise ValueError(f"{source}: snippet region end without matching start: {name!r}")
        else:
            active_name, start = active
            if active_name != name:
                raise ValueError(
                    f"{source}: reversed or overlapping snippet regions: "
                    f"expected end for {active_name!r}, got {name!r}"
                )
            content = "".join(lines[start + 1 : index])
            if not content.strip():
                raise ValueError(f"{source}: empty snippet region {name!r}")
            regions[name] = content
            active = None
    if active is not None:
        raise ValueError(f"{source}: snippet region missing end: {active[0]!r}")
    return regions


def snippet_source(root: pathlib.Path, doc: pathlib.Path, source_rel: str) -> str:
    source_name, separator, region_name = source_rel.partition("#")
    if not source_name:
        raise ValueError(f"{doc}: snippet source path is empty: {source_rel}")
    source_path = pathlib.Path(source_name)
    if source_path.is_absolute():
        raise ValueError(
            f"{doc}: snippet source must be relative to repository root: {source_rel}"
        )
    source = (root / source_path).resolve()
    try:
        source.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"{doc}: snippet source escapes repository root: {source_rel}"
        ) from exc
    if not source.is_file():
        raise ValueError(f"{doc}: snippet source does not exist: {source_rel}")
    if not separator:
        return source.read_text(encoding="utf-8")
    if not region_name:
        raise ValueError(f"{doc}: snippet region name is empty: {source_rel}")
    regions = source_regions(source)
    try:
        return regions[region_name]
    except KeyError as exc:
        raise ValueError(f"{doc}: unknown snippet region {region_name!r} in {source_rel}") from exc


def fenced(source: str) -> str:
    return "```rust\n" + source.rstrip() + "\n```\n"


def canonical_skill_docs(root: pathlib.Path) -> list[pathlib.Path]:
    skill_root = root / ".agents" / "skills" / "tenferro-compute"
    if not skill_root.is_dir():
        return []
    return sorted(skill_root.rglob("*.md"))


def unmarked_rust_fences(root: pathlib.Path) -> list[str]:
    inventory: list[str] = []
    docs = [
        doc
        for directory in (root / "docs" / "guides", root / "docs" / "tutorials")
        for doc in sorted(directory.glob("*.md"))
    ]
    docs.extend(canonical_skill_docs(root))
    for doc in docs:
        marked = False
        for line_number, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), 1):
            if START_RE.search(line):
                marked = True
            elif END_RE.search(line):
                marked = False
            elif line.strip() == "```rust" and not marked:
                inventory.append(f"{doc.relative_to(root)}:{line_number}")
    return inventory


def rewrite_doc(root: pathlib.Path, doc: pathlib.Path) -> tuple[str, bool]:
    text = doc.read_text()
    out: list[str] = []
    pos = 0
    changed = False
    while True:
        start = START_RE.search(text, pos)
        next_end = END_RE.search(text, pos)
        if next_end and (not start or next_end.start() < start.start()):
            raise ValueError(
                f"{doc}: end-snippet-source marker without matching snippet-source marker"
            )
        if not start:
            out.append(text[pos:])
            break
        end = END_RE.search(text, start.end())
        if not end:
            raise ValueError(f"{doc}: missing end-snippet-source marker")
        source_rel = start.group(1).strip()
        source_text = snippet_source(root, doc, source_rel)
        replacement = (
            text[start.start() : start.end()]
            + "\n"
            + fenced(source_text)
            + text[end.start() : end.end()]
        )
        current = text[start.start() : end.end()]
        out.append(text[pos : start.start()])
        out.append(replacement)
        changed = changed or current != replacement
        pos = end.end()
    return "".join(out), changed


def user_facing_docs(root: pathlib.Path) -> list[pathlib.Path]:
    docs_root = root / "docs"
    excluded_parts = {
        "plans",
        "superpowers",
        "design",
        "architecture",
        "spec",
        "reference",
        "oracle",
    }
    docs: list[pathlib.Path] = [root / "README.md"]
    for path in sorted(docs_root.rglob("*.md")):
        relative = path.relative_to(docs_root)
        if relative.parts and relative.parts[0] in excluded_parts:
            continue
        docs.append(path)
    docs.extend(canonical_skill_docs(root))
    return docs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", default=pathlib.Path(__file__).resolve().parents[1])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    root = pathlib.Path(args.root_dir).resolve()
    changed_docs: list[pathlib.Path] = []
    try:
        for doc in user_facing_docs(root):
            new_text, changed = rewrite_doc(root, doc)
            if changed:
                changed_docs.append(doc)
                if not args.check:
                    doc.write_text(new_text)
        unmarked = unmarked_rust_fences(root)
        if args.check and unmarked:
            print(
                f"unmarked plain Rust fences ({len(unmarked)}):",
                file=sys.stderr,
            )
            for fence in unmarked:
                print(f"- {fence}", file=sys.stderr)
            return 1
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if changed_docs and args.check:
        print("stale doc snippets:", file=sys.stderr)
        for doc in changed_docs:
            print(f"- {doc.relative_to(root)}", file=sys.stderr)
        print("run: python3 scripts/check-doc-snippets.py", file=sys.stderr)
        return 1

    if changed_docs:
        print(f"updated {len(changed_docs)} doc snippet file(s)")
    else:
        print("doc-snippets-ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
