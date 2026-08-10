#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
from html.parser import HTMLParser
from urllib.parse import unquote, urlsplit

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python < 3.11
    tomllib = None


LLMS_SITE_PREFIX = "/tenferro-rs/"
LLMS_LINK_RE = re.compile(r"^\s*-\s+\[([^]]+)\]\(([^)]+)\):\s*(\S.*)$", re.MULTILINE)
LLMS_SKILL_PATH = ".agents/skills/tenferro-compute/SKILL.md"
LLMS_SKILL_REFERENCES_PREFIX = "skill-references/"
LLMS_README_URL = "https://github.com/tensor4all/tenferro-rs/blob/main/README.md"


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


class RenderedPageLinkCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        href = dict(attrs).get("href") or ""
        if href:
            self.links.add(href)


def rendered_page_links(path: pathlib.Path) -> set[str]:
    parser = RenderedPageLinkCollector()
    parser.feed(path.read_text(encoding="utf-8"))
    return parser.links


def is_external_href(href: str) -> bool:
    parsed = urlsplit(href)
    if parsed.scheme or parsed.netloc:
        return True
    return href.startswith("#") or href.startswith("/")


def rendered_html_pages(site_root: pathlib.Path) -> list[pathlib.Path]:
    if not site_root.exists():
        return []
    pages: list[pathlib.Path] = []
    for path in sorted(site_root.rglob("*.html")):
        try:
            relative = path.relative_to(site_root)
        except ValueError:
            continue
        if relative.parts and relative.parts[0] == "api":
            continue
        pages.append(path)
    return pages


def missing_rendered_html_links(site_root: pathlib.Path) -> list[tuple[pathlib.Path, str, pathlib.Path]]:
    # This check intentionally starts with rendered page-to-page links. Asset
    # link validation can be added here once docs-site asset ownership settles.
    missing: list[tuple[pathlib.Path, str, pathlib.Path]] = []
    for page in rendered_html_pages(site_root):
        for href in sorted(rendered_page_links(page)):
            if is_external_href(href):
                continue
            parsed = urlsplit(href)
            if not parsed.path or not parsed.path.endswith(".html"):
                continue
            target = (page.parent / unquote(parsed.path)).resolve()
            try:
                target.relative_to(site_root)
            except ValueError:
                continue
            if not target.is_file():
                missing.append((page, href, target))
    return missing


def llms_source_path(root: pathlib.Path, url: str) -> pathlib.Path | None:
    parsed = urlsplit(url)
    if parsed.netloc == "tensor4all.org" and parsed.path.startswith(LLMS_SITE_PREFIX):
        relative = unquote(parsed.path[len(LLMS_SITE_PREFIX) :])
        # Republished skill references are copied into the site build from the
        # canonical skill; resolve them back to that single source of truth.
        if relative.startswith(LLMS_SKILL_REFERENCES_PREFIX):
            name = relative[len(LLMS_SKILL_REFERENCES_PREFIX) :]
            if not name.endswith(".md"):
                return None
            return root / ".agents" / "skills" / "tenferro-compute" / "references" / name
        if relative.endswith("/"):
            relative += "index.md"
        elif relative.endswith(".html"):
            relative = relative[:-5] + ".md"
        else:
            return None
        return root / "docs" / relative
    github_prefix = "/tensor4all/tenferro-rs/blob/main/"
    if parsed.netloc == "github.com" and parsed.path.startswith(github_prefix):
        return root / unquote(parsed.path[len(github_prefix) :])
    return None


def llms_skill_reference_names(text: str) -> set[str]:
    names: set[str] = set()
    for match in LLMS_LINK_RE.finditer(text):
        url = match.group(2)
        parsed = urlsplit(url)
        if parsed.netloc != "tensor4all.org":
            continue
        relative = unquote(parsed.path)
        if not relative.startswith(LLMS_SITE_PREFIX + LLMS_SKILL_REFERENCES_PREFIX):
            continue
        name = relative[len(LLMS_SITE_PREFIX + LLMS_SKILL_REFERENCES_PREFIX) :]
        if name.endswith(".md"):
            names.add(name)
    return names


def markdown_link_targets(text: str) -> set[str]:
    return {target for target in re.findall(r"\[[^]]*\]\(([^)]+)\)", text)}


def check_reachability(root: pathlib.Path) -> list[str]:
    """Assert the README is the single router and llms.txt links back to it."""
    errors: list[str] = []
    readme = root / "README.md"
    if not readme.is_file():
        return ["README.md is missing; router reachability cannot be checked"]
    readme_targets = markdown_link_targets(readme.read_text(encoding="utf-8"))
    if not any(target == "docs/llms.txt" or target.endswith("/docs/llms.txt") for target in readme_targets):
        errors.append("README.md must link docs/llms.txt (single router)")
    if not any(LLMS_SKILL_PATH in target for target in readme_targets):
        errors.append(f"README.md must link {LLMS_SKILL_PATH}")
    llms_index = root / "docs" / "llms.txt"
    llms_targets = (
        markdown_link_targets(llms_index.read_text(encoding="utf-8")) if llms_index.is_file() else set()
    )
    if LLMS_README_URL not in llms_targets:
        errors.append(f"docs/llms.txt must link back to the README ({LLMS_README_URL})")
    return errors


def check_llms_index(root: pathlib.Path, docs_site_root: pathlib.Path | None = None) -> list[str]:
    index = root / "docs" / "llms.txt"
    if not index.is_file():
        return ["docs/llms.txt is missing"]

    quarto = root / "docs" / "_quarto.yml"
    if not quarto.is_file() or not re.search(r"(?m)^\s*-\s*llms\.txt\s*$", quarto.read_text(encoding="utf-8")):
        return ["docs/_quarto.yml must list llms.txt under project resources"]

    errors: list[str] = []
    seen_urls: set[str] = set()
    matches = list(LLMS_LINK_RE.finditer(index.read_text(encoding="utf-8")))
    if not matches:
        return ["docs/llms.txt has no described Markdown links"]
    for match in matches:
        label, url, description = match.groups()
        if url in seen_urls:
            errors.append(f"docs/llms.txt repeats URL: {url}")
        seen_urls.add(url)
        if not description.strip():
            errors.append(f"docs/llms.txt has an empty description for: {label}")
        target = llms_source_path(root, url)
        if target is None:
            errors.append(f"docs/llms.txt has an unsupported URL: {url}")
        elif not target.is_file():
            errors.append(f"docs/llms.txt target does not exist: {url} -> {target.relative_to(root)}")

    if not any(url.endswith(LLMS_SKILL_PATH) for url in seen_urls):
        errors.append(f"docs/llms.txt must link {LLMS_SKILL_PATH}")
    elif not (root / LLMS_SKILL_PATH).is_file():
        errors.append(f"docs/llms.txt skill target does not exist: {LLMS_SKILL_PATH}")
    if docs_site_root is not None and docs_site_root.exists():
        if not (docs_site_root / "llms.txt").is_file():
            errors.append(f"built docs site is missing root llms.txt: {docs_site_root / 'llms.txt'}")
        for name in sorted(llms_skill_reference_names(index.read_text(encoding="utf-8"))):
            built = docs_site_root / LLMS_SKILL_REFERENCES_PREFIX / name
            if not built.is_file():
                errors.append(f"built docs site is missing republished skill reference: {built}")
    return errors


def check_eager_functional_ad_docs(root: pathlib.Path) -> list[str]:
    required_snippets = [
        (
            root / "README.md",
            "Both eager and traced modes support VJP and JVP",
        ),
        (
            root / "README.md",
            "HVP-style higher-order composition",
        ),
        (
            root / "docs" / "index.md",
            "`EagerRuntime` functional `grad`, `vjp`, and `jvp`",
        ),
        (
            root / "docs" / "getting-started" / "index.md",
            "functional eager `grad`, `vjp`, and `jvp`",
        ),
        (
            root / "docs" / "getting-started" / "core-concepts.md",
            "functional `grad`, `vjp`, and `jvp` transforms",
        ),
        (
            root / "docs" / "getting-started" / "pytorch-jax-mapping.md",
            "`EagerRuntime` functional `grad`/`vjp`/`jvp`",
        ),
        (
            root / "docs" / "tutorials" / "index.md",
            "functional eager AD entry point",
        ),
        (
            root / "docs" / "spec" / "operation-categories.md",
            "stateful `backward()` plus functional `grad`/`vjp`/`jvp`",
        ),
        (
            root / "docs" / "guides" / "eager-operations.md",
            "stateful reverse-mode and functional `grad`/`vjp`/`jvp`",
        ),
        (
            root / "docs" / "assets" / "tenferro-architecture.svg",
            "backward · grad",
        ),
        (
            root / "docs" / "assets" / "tenferro-architecture.svg",
            "vjp · jvp",
        ),
    ]
    missing: list[str] = []
    for path, snippet in required_snippets:
        text = path.read_text(encoding="utf-8")
        normalized_text = " ".join(text.split())
        normalized_snippet = " ".join(snippet.split())
        if normalized_snippet not in normalized_text:
            missing.append(f"{path.relative_to(root)}: missing {snippet!r}")
    return missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify docs-site completeness for workspace library crates.")
    parser.add_argument("--root-dir", default=".", help="Repository root (default: current directory)")
    parser.add_argument("--doc-root", help="Rustdoc output directory (default: <root>/target/doc)")
    parser.add_argument("--api-index-md", help="Markdown API index (default: <root>/docs/api/index.md if it exists)")
    parser.add_argument("--site-index", help="Rendered API landing page HTML (default: <root>/target/docs-site/api/index.html if it exists)")
    parser.add_argument("--docs-site-root", help="Rendered Quarto site root (default: <root>/target/docs-site)")
    parser.add_argument("--quiet", action="store_true", help="Suppress success output")
    return parser.parse_args()


def load_workspace_libs(root: pathlib.Path) -> list[tuple[str, str, str]]:
    if tomllib is None:
        metadata = subprocess.run(
            ["cargo", "metadata", "--no-deps", "--format-version", "1"],
            check=False,
            cwd=root,
            stdout=subprocess.PIPE,
            text=True,
        )
        if metadata.returncode != 0:
            raise RuntimeError("cargo metadata failed while discovering workspace crates")
        workspace = json.loads(metadata.stdout)
        members = set(workspace["workspace_members"])
        crates: list[tuple[str, str, str]] = []
        for package in workspace["packages"]:
            if package["id"] not in members:
                continue
            lib_targets = [
                target
                for target in package["targets"]
                if "lib" in target["kind"] or "proc-macro" in target["kind"]
            ]
            if not lib_targets:
                continue
            member = pathlib.Path(package["manifest_path"]).parent
            crates.append(
                (
                    str(member.relative_to(root)),
                    package["name"],
                    lib_targets[0]["name"],
                )
            )
        return crates

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
        lib_name = manifest.get("lib", {}).get("name", package_name.replace("-", "_"))
        crates.append((member, package_name, lib_name))
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
    snippet_check = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "check-doc-snippets.py"),
            "--root-dir",
            str(root),
            "--check",
        ],
        check=False,
        stdout=subprocess.DEVNULL if args.quiet else None,
    )
    if snippet_check.returncode != 0:
        return snippet_check.returncode
    guide_dependency_check = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "check-guide-dependency-snippets.py"),
            "--root-dir",
            str(root),
        ],
        check=False,
        stdout=subprocess.DEVNULL if args.quiet else None,
    )
    if guide_dependency_check.returncode != 0:
        return guide_dependency_check.returncode

    doc_root = pathlib.Path(args.doc_root) if args.doc_root else root / "target" / "doc"
    api_index_md = pathlib.Path(args.api_index_md) if args.api_index_md else root / "docs" / "api" / "index.md"
    site_index = pathlib.Path(args.site_index) if args.site_index else root / "target" / "docs-site" / "api" / "index.html"
    docs_site_root = (
        pathlib.Path(args.docs_site_root).resolve()
        if args.docs_site_root
        else root / "target" / "docs-site"
    )

    llms_errors = check_llms_index(root, docs_site_root)
    if llms_errors:
        print("llms.txt validation failed:", file=sys.stderr)
        for error in llms_errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    reachability_errors = check_reachability(root)
    if reachability_errors:
        print("docs reachability failed:", file=sys.stderr)
        for error in reachability_errors:
            print(f"- {error}", file=sys.stderr)
        return 1

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

    missing_site_links = missing_rendered_html_links(docs_site_root)
    if missing_site_links:
        print("rendered docs links outside the rendered docs set:", file=sys.stderr)
        for source, href, target in missing_site_links:
            source_rel = source.relative_to(docs_site_root)
            target_rel = target.relative_to(docs_site_root)
            print(f"- {source_rel}: {href} -> {target_rel}", file=sys.stderr)
        return 1

    eager_ad_doc_gaps = check_eager_functional_ad_docs(root)
    if eager_ad_doc_gaps:
        print("eager functional AD docs are stale:", file=sys.stderr)
        for gap in eager_ad_doc_gaps:
            print(f"- {gap}", file=sys.stderr)
        return 1

    if not args.quiet:
        print(f"docs-site-ok: {len(crates)} workspace library crates verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
