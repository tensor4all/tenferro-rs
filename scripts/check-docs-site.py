#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import re
import sys
import tomllib
from html.parser import HTMLParser


class LinkCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        href = dict(attrs).get("href") or ""
        match = re.search(r"(?:^|/)([A-Za-z0-9_\-]+)/index\.html$", href)
        if match:
            self.links.add(match.group(1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify docs-site completeness for workspace library crates.")
    parser.add_argument("--root-dir", default=".", help="Repository root (default: current directory)")
    parser.add_argument("--doc-root", help="Rustdoc output directory (default: <root>/target/doc)")
    parser.add_argument("--api-index-md", help="Markdown API index (default: <root>/docs/api_index.md if it exists)")
    parser.add_argument("--site-index", help="Rendered API landing page HTML (default: <root>/target/docs-site/api/index.html if it exists)")
    parser.add_argument("--quiet", action="store_true", help="Suppress success output")
    return parser.parse_args()


def load_workspace_libs(root: pathlib.Path) -> list[tuple[str, str, str]]:
    with (root / "Cargo.toml").open("rb") as handle:
        workspace = tomllib.load(handle)["workspace"]

    crates: list[tuple[str, str, str]] = []
    for member in workspace["members"]:
        member_path = root / member
        with (member_path / "Cargo.toml").open("rb") as handle:
            manifest = tomllib.load(handle)
        if "package" not in manifest:
            continue
        if "lib" not in manifest and not (member_path / "src" / "lib.rs").exists():
            continue
        package_name = manifest["package"]["name"]
        crates.append((member, package_name, package_name.replace("-", "_")))
    return crates


def markdown_links(path: pathlib.Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    return set(re.findall(r"\((?:\./)?([A-Za-z0-9_\-]+)/index\.html\)", text))


def html_links(path: pathlib.Path) -> set[str]:
    parser = LinkCollector()
    parser.feed(path.read_text(encoding="utf-8"))
    return parser.links


def main() -> int:
    args = parse_args()
    root = pathlib.Path(args.root_dir).resolve()
    doc_root = pathlib.Path(args.doc_root) if args.doc_root else root / "target" / "doc"
    api_index_md = pathlib.Path(args.api_index_md) if args.api_index_md else root / "docs" / "api_index.md"
    site_index = pathlib.Path(args.site_index) if args.site_index else root / "target" / "docs-site" / "api" / "index.html"

    crates = load_workspace_libs(root)
    missing_doc = [pkg for _member, pkg, doc_dir in crates if not (doc_root / doc_dir / "index.html").exists()]
    if missing_doc:
        print("missing rustdoc output for:", file=sys.stderr)
        for pkg in missing_doc:
            print(f"- {pkg}", file=sys.stderr)
        return 1

    linked_dirs: set[str] | None = None
    link_source: pathlib.Path | None = None
    if site_index.exists():
        linked_dirs = html_links(site_index)
        link_source = site_index
    elif api_index_md.exists():
        linked_dirs = markdown_links(api_index_md)
        link_source = api_index_md

    if linked_dirs is not None:
        missing_links = [pkg for _member, pkg, doc_dir in crates if doc_dir not in linked_dirs]
        if missing_links:
            print(f"missing crate links in {link_source}:", file=sys.stderr)
            for pkg in missing_links:
                print(f"- {pkg}", file=sys.stderr)
            return 1

    if not args.quiet:
        print(f"docs-site-ok: {len(crates)} workspace library crates verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
