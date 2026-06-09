#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import re
import sys

START_RE = re.compile(r"<!--\s*snippet-source:\s*([^>]+?)\s*-->")
END_RE = re.compile(r"<!--\s*end-snippet-source\s*-->")


def fenced(source: pathlib.Path) -> str:
    return "```rust\n" + source.read_text().rstrip() + "\n```\n"


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
        source_path = pathlib.Path(source_rel)
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
        replacement = (
            text[start.start() : start.end()]
            + "\n"
            + fenced(source)
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
