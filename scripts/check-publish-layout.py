#!/usr/bin/env python3
"""Check tenferro publish layout and crates.io-facing metadata.

This check allows pre-publish git dependencies only when they also declare the
registry version Cargo needs for packaging.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
USER_CRATE_ORDER = [
    "tenferro-tensor",
    "tenferro-cpu",
    "tenferro-gpu",
    "tenferro-runtime",
    "tenferro-ad",
    "tenferro-xla",
]
EXTENSION_CRATE_ORDER = [
    "tenferro-linalg",
    "tenferro-einsum",
    "tenferro-fft",
]
IMPLEMENTATION_CRATE_ORDER = [
    "tenferro-tensor-core",
    "tenferro-core-ops",
    "tenferro-internal-cpu-kernels",
    "tenferro-internal-ops",
    "tenferro-internal-extension-macros",
]
TENFERRO_CRATES = (
    USER_CRATE_ORDER + EXTENSION_CRATE_ORDER + IMPLEMENTATION_CRATE_ORDER
)
VALID_CATEGORIES = {
    "algorithms",
    "api-bindings",
    "data-structures",
    "development-tools",
    "development-tools::procedural-macro-helpers",
    "external-ffi-bindings",
    "hardware-support",
    "mathematics",
    "no-std",
    "rust-patterns",
    "science",
    "simulation",
}


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def cargo_metadata() -> dict:
    completed = subprocess.run(
        ["cargo", "metadata", "--no-deps", "--format-version", "1"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return json.loads(completed.stdout)


def section(text: str, heading: str) -> str:
    marker = f"[{heading}]"
    start = text.find(marker)
    if start < 0:
        return ""
    rest = text[start + len(marker) :]
    next_section = rest.find("\n[")
    if next_section >= 0:
        return rest[:next_section]
    return rest


def markdown_section(text: str, heading: str) -> str:
    marker = f"## {heading}"
    start = text.find(marker)
    if start < 0:
        return ""
    rest = text[start + len(marker) :]
    next_section = rest.find("\n## ")
    if next_section >= 0:
        return rest[:next_section]
    return rest


def publishable_crates(metadata: dict) -> set[str]:
    workspace_members = set(metadata["workspace_members"])
    return {
        package["name"]
        for package in metadata["packages"]
        if package["id"] in workspace_members and package.get("publish") != []
    }


def phase3_publication_order(
    release_text: str, errors: list[str]
) -> list[str] | None:
    heading = "## Phase 3 — Publish From The Tag"
    lines = release_text.splitlines()
    headings: list[int] = []
    active_marker: tuple[str, int] | None = None
    for index, line in enumerate(lines):
        fence = re.match(r"^(`{3,}|~{3,})(.*)$", line.strip())
        if active_marker is not None:
            marker_char, marker_len = active_marker
            if (
                fence
                and fence.group(1)[0] == marker_char
                and len(fence.group(1)) >= marker_len
                and not fence.group(2).strip()
            ):
                active_marker = None
        elif fence:
            marker = fence.group(1)
            active_marker = (marker[0], len(marker))
        elif re.fullmatch(rf" {{0,3}}{re.escape(heading)}", line):
            headings.append(index)

    if len(headings) != 1:
        errors.append("release workflow must contain exactly one exact Phase 3 heading")
        return None

    text_fences: list[list[str]] = []
    active_fence: tuple[str, int, str, list[str]] | None = None
    for line in lines[headings[0] + 1 :]:
        fence = re.match(r"^(`{3,}|~{3,})(.*)$", line.strip())
        if active_fence is None:
            if re.match(r"^ {0,3}#{1,6}(?:[ \t]+|$)", line):
                break
            if fence:
                marker, info = fence.groups()
                active_fence = (marker[0], len(marker), info.strip(), [])
            continue

        marker_char, marker_len, info, contents = active_fence
        if (
            fence
            and fence.group(1)[0] == marker_char
            and len(fence.group(1)) >= marker_len
            and not fence.group(2).strip()
        ):
            if info == "text":
                text_fences.append(contents)
            active_fence = None
        else:
            contents.append(line)

    if active_fence is not None or len(text_fences) != 1:
        errors.append(
            "release workflow Phase 3 must contain exactly one complete text fence"
        )
        return None
    return [line.strip() for line in text_fences[0] if line.strip()]


def check_release_order(
    metadata: dict, release_text: str, errors: list[str]
) -> None:
    order = phase3_publication_order(release_text, errors)
    if order is None:
        return
    expected = publishable_crates(metadata)
    if len(order) != len(set(order)):
        errors.append("release publish order must not contain duplicate crates")
    missing = sorted(expected - set(order))
    unexpected = sorted(set(order) - expected)
    if missing:
        errors.append(f"release publish order is missing crates: {missing}")
    if unexpected:
        errors.append(f"release publish order has unexpected crates: {unexpected}")
    if missing or unexpected or len(order) != len(set(order)):
        return

    positions = {crate: index for index, crate in enumerate(order)}
    for package in metadata["packages"]:
        crate = package["name"]
        if crate not in expected:
            continue
        # A VERSIONED normal, build, or dev dependency on a publishable crate
        # that appears later in the canonical order cannot resolve during
        # `cargo package`, because that crate has not been published yet.
        # Unversioned (path-only) dev-dependencies are safe: dev-dependencies
        # are stripped from the published manifest and never carry a registry
        # version requirement for consumers.
        for dependency in package["dependencies"]:
            dependency_name = dependency["name"]
            if dependency_name not in expected:
                continue
            if _is_versioned(dependency) and positions[dependency_name] > positions[crate]:
                errors.append(
                    f"release publish order must place dependency "
                    f"{dependency_name!r} before {crate!r}"
                )

    cycle = _find_publication_cycle(metadata["packages"], expected)
    if cycle is not None:
        chain = " -> ".join(cycle + [cycle[0]])
        errors.append(f"publication cycle among publishable crates: {chain}")


def _is_versioned(dependency: dict) -> bool:
    """True when the dependency carries a cargo version requirement.

    Path-only manifests (no `version`) report `req == "*"` from cargo
    metadata. Only versioned edges are published across crates and therefore
    participate in publish-order and publish-cycle constraints.
    """

    requirement = dependency.get("req")
    return isinstance(requirement, str) and requirement != "*"


def _find_publication_cycle(
    packages: list[dict], publishable: set[str]
) -> list[str] | None:
    """Return one cycle over publishable crate names, or None when acyclic.

    Only VERSIONED edges participate: an unversioned dev-dependency is
    stripped from the published manifest and never needs crates.io resolution,
    so it cannot create a publish-time cycle. Any versioned cycle forces at
    least one edge to point forward in any linear publication order, so it is
    reported as a structural publish-cycle before tagging.
    """

    adjacency = {
        package["name"]: sorted(
            dependency["name"]
            for dependency in package["dependencies"]
            if dependency["name"] in publishable and _is_versioned(dependency)
        )
        for package in packages
        if package["name"] in publishable
    }

    WHITE, GRAY, BLACK = 0, 1, 2
    color = {crate: WHITE for crate in adjacency}
    stack: list[str] = []

    def visit(crate: str) -> list[str] | None:
        color[crate] = GRAY
        stack.append(crate)
        for neighbor in adjacency[crate]:
            if color[neighbor] == GRAY:
                return stack[stack.index(neighbor) :]
            if color[neighbor] == WHITE:
                cycle = visit(neighbor)
                if cycle is not None:
                    return cycle
        stack.pop()
        color[crate] = BLACK
        return None

    for crate in sorted(adjacency):
        if color[crate] == WHITE:
            cycle = visit(crate)
            if cycle is not None:
                return cycle
    return None


def check_workspace_members(metadata: dict, root_text: str, errors: list[str]) -> None:
    packages = {package["name"]: package for package in metadata["packages"]}

    if "tenferro" in packages:
        errors.append("workspace must not include a root tenferro facade crate")

    workspace_crates = publishable_crates(metadata)
    expected_crates = set(TENFERRO_CRATES)
    missing_crates = sorted(expected_crates - workspace_crates)
    unexpected_crates = sorted(workspace_crates - expected_crates)
    if missing_crates:
        errors.append(f"workspace missing published crates: {missing_crates}")
    if unexpected_crates:
        errors.append(f"workspace has unexpected published crates: {unexpected_crates}")

    for crate in TENFERRO_CRATES:
        package = packages.get(crate)
        if package is None:
            errors.append(f"workspace missing package {crate!r}")
            continue
        manifest_path = Path(package["manifest_path"])
        expected = ROOT / "crates" / crate / "Cargo.toml"
        if manifest_path != expected:
            errors.append(
                f"{crate} manifest must be {rel(expected)}, found {rel(manifest_path)}"
            )

    workspace_section = section(root_text, "workspace")
    if '"crates/tenferro-tensor"' not in workspace_section:
        errors.append("workspace members must include 'crates/tenferro-tensor'")
    if '"tenferro-tensor"' in workspace_section:
        errors.append("workspace members/default-members must not use top-level tenferro-tensor")
    if '"crates/tenferro-tensor",' not in root_text:
        errors.append(
            "workspace default-members must be exactly ['crates/tenferro-tensor']"
        )


def check_workspace_metadata(root_text: str, errors: list[str]) -> None:
    package_section = section(root_text, "workspace.package")
    required_lines = [
        'rust-version = "1.96"',
        'publish = true',
        'readme = "README.md"',
        'repository = "https://github.com/tensor4all/tenferro-rs"',
        'homepage = "https://tensor4all.org/tenferro-rs/"',
    ]
    for line in required_lines:
        if line not in package_section:
            errors.append(f"workspace.package missing {line!r}")


def check_workspace_dependencies(root_text: str, errors: list[str]) -> None:
    try:
        manifest = tomllib.loads(root_text)
    except tomllib.TOMLDecodeError as error:
        errors.append(f"Cargo.toml has invalid TOML: {error}")
        return
    dependencies = manifest.get("workspace", {}).get("dependencies", {})
    if not isinstance(dependencies, dict):
        errors.append("Cargo.toml [workspace.dependencies] must be a table")
        return
    for name, dependency in dependencies.items():
        if not isinstance(dependency, dict) or "git" not in dependency:
            continue
        for key in ("version", "rev"):
            if not isinstance(dependency.get(key), str) or not dependency[key]:
                errors.append(
                    f"Cargo.toml workspace git dependency {name!r} must include "
                    f"a {key} for crates.io packaging"
                )


def workspace_version(root_text: str) -> str:
    package_section = section(root_text, "workspace.package")
    for line in package_section.splitlines():
        stripped = line.strip()
        if stripped.startswith("version = "):
            return stripped.split("=", 1)[1].strip().strip('"')
    raise ValueError("workspace.package missing version")


def check_crate_metadata(
    crate: str, manifest_text: str, errors: list[str], manifest_name: str
) -> None:
    try:
        manifest = tomllib.loads(manifest_text)
    except tomllib.TOMLDecodeError as error:
        errors.append(f"{manifest_name} has invalid TOML: {error}")
        return

    package = manifest.get("package", {})
    if package.get("rust-version") != {"workspace": True}:
        errors.append(
            f"{manifest_name} package.rust-version must inherit workspace metadata"
        )

    for key in ("keywords", "categories"):
        values = package.get(key)
        if not isinstance(values, list) or not 1 <= len(values) <= 5:
            errors.append(f"{manifest_name} package.{key} must contain 1-5 entries")
        elif any(not isinstance(value, str) or not value for value in values):
            errors.append(f"{manifest_name} package.{key} entries must be non-empty strings")
        elif key == "keywords":
            if len(set(values)) != len(values):
                errors.append(f"{manifest_name} package.keywords must be unique")
            invalid_keywords = [
                value
                for value in values
                if len(value) > 20
                or not value[0].isascii()
                or not value[0].isalnum()
                or any(
                    not character.isascii()
                    or not (character.isalnum() or character in "_+-")
                    for character in value[1:]
                )
            ]
            if invalid_keywords:
                errors.append(
                    f"{manifest_name} package.keywords contain invalid crates.io syntax: "
                    f"{invalid_keywords}"
                )

    categories = package.get("categories")
    if isinstance(categories, list):
        invalid = sorted(set(categories) - VALID_CATEGORIES)
        if invalid:
            errors.append(
                f"{manifest_name} package.categories has invalid crates.io slugs: {invalid}"
            )

    expected_documentation = f"https://docs.rs/{crate}"
    if package.get("documentation") != expected_documentation:
        errors.append(
            f"{manifest_name} package.documentation must be {expected_documentation!r}"
        )

    docs_rs = package.get("metadata", {}).get("docs", {}).get("rs")
    if not isinstance(docs_rs, dict):
        errors.append(f"{manifest_name} missing [package.metadata.docs.rs]")
        return
    if docs_rs.get("rustdoc-args") != ["--cfg", "docsrs"]:
        errors.append(
            f"{manifest_name} docs.rs rustdoc-args must be ['--cfg', 'docsrs']"
        )

    has_all_features = "all-features" in docs_rs
    has_explicit_features = "features" in docs_rs
    if has_all_features == has_explicit_features:
        errors.append(
            f"{manifest_name} docs.rs must set exactly one of all-features or features"
        )
    elif has_all_features:
        if docs_rs["all-features"] is not True:
            errors.append(f"{manifest_name} docs.rs all-features must be true")
    else:
        features = docs_rs["features"]
        if not isinstance(features, list) or not features:
            errors.append(f"{manifest_name} docs.rs features must be a non-empty list")
        elif any(not isinstance(feature, str) or not feature for feature in features):
            errors.append(f"{manifest_name} docs.rs features must be non-empty strings")
        else:
            defined_features = manifest.get("features", {})
            if not isinstance(defined_features, dict):
                defined_features = {}
            missing_features = sorted(set(features) - set(defined_features))
            if missing_features:
                errors.append(
                    f"{manifest_name} docs.rs features are not defined: {missing_features}"
                )


def _check_internal_dependency_versions(
    crate: str,
    manifest_text: str,
    expected_version: str,
    errors: list[str],
    manifest_name: str,
) -> None:
    """Require registry versions on internal normal/build path dependencies.

    Normal and build dependencies are kept in the published manifest, so they
    must carry the exact workspace version for crates.io packaging. Dev
    dependencies are stripped from the published manifest and never resolve
    against crates.io for consumers, so they may be path-only (unversioned);
    when a dev dependency does declare a version it must still match.
    """

    try:
        manifest = tomllib.loads(manifest_text)
    except tomllib.TOMLDecodeError:
        return

    version_fragment = f'version = "{expected_version}"'
    # Normal and build dependencies are kept in the published manifest.
    for table_name in ("dependencies", "build-dependencies"):
        table = manifest.get(table_name, {})
        if not isinstance(table, dict):
            continue
        for name, spec in table.items():
            if not isinstance(spec, dict) or "path" not in spec:
                continue
            if not spec["path"].startswith("../tenferro-"):
                continue
            if spec.get("version") != expected_version:
                errors.append(
                    f"{manifest_name} {table_name} tenferro dependency "
                    f"{name!r} must include {version_fragment} for crates.io packaging"
                )
    # Dev dependencies are stripped from the published manifest, so they may
    # be path-only; any declared version must still match.
    table = manifest.get("dev-dependencies", {})
    if isinstance(table, dict):
        for name, spec in table.items():
            if not isinstance(spec, dict) or "path" not in spec:
                continue
            if not spec["path"].startswith("../tenferro-"):
                continue
            declared = spec.get("version")
            if declared is not None and declared != expected_version:
                errors.append(
                    f"{manifest_name} dev-dependencies tenferro dependency "
                    f"{name!r} must include {version_fragment} for crates.io packaging"
                )


def check_package_metadata(root_text: str, errors: list[str]) -> None:
    expected_version = workspace_version(root_text)
    for crate in TENFERRO_CRATES:
        manifest_path = ROOT / "crates" / crate / "Cargo.toml"
        if not manifest_path.exists():
            errors.append(f"missing manifest {rel(manifest_path)}")
            continue

        manifest_text = manifest_path.read_text(encoding="utf-8")
        package_section = section(manifest_text, "package")
        if "publish.workspace = true" not in package_section:
            errors.append(
                f"{rel(manifest_path)} package.publish must inherit workspace metadata"
            )

        for key in ("readme", "repository", "homepage"):
            if f"{key}.workspace = true" not in package_section:
                errors.append(
                    f"{rel(manifest_path)} package.{key} must inherit workspace metadata"
                )
        check_crate_metadata(crate, manifest_text, errors, rel(manifest_path))
        _check_internal_dependency_versions(
            crate, manifest_text, expected_version, errors, rel(manifest_path)
        )

    tutorial_manifest = ROOT / "docs" / "tutorial-code" / "Cargo.toml"
    if "publish = false" not in section(
        tutorial_manifest.read_text(encoding="utf-8"), "package"
    ):
        errors.append("docs/tutorial-code must remain publish = false")

    for extension in ("tropical", "sparse", "tenferro-cpu-tblis"):
        extension_manifest = ROOT / "ext" / extension / "Cargo.toml"
        if "publish = false" not in section(
            extension_manifest.read_text(encoding="utf-8"), "package"
        ):
            errors.append(f"ext/{extension} must remain publish = false")


def check_readme(errors: list[str]) -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    required_fragments = [
        "There is intentionally no `tenferro` facade crate",
        "### Core User Crates",
        "### Standard Operation Extensions",
        "### Implementation Crates",
    ]
    for fragment in required_fragments:
        if fragment not in readme:
            errors.append(f"README.md missing fragment: {fragment!r}")

    crates_section = markdown_section(readme, "Crates")
    if not crates_section:
        errors.append("README.md missing ## Crates section")
        crates_section = readme

    ordered = USER_CRATE_ORDER + EXTENSION_CRATE_ORDER + IMPLEMENTATION_CRATE_ORDER
    positions = []
    for crate in ordered:
        marker = f"`{crate}`"
        pos = crates_section.find(marker)
        if pos < 0:
            errors.append(f"README.md missing crate entry {marker}")
        positions.append(pos)

    present_positions = [pos for pos in positions if pos >= 0]
    if present_positions != sorted(present_positions):
        errors.append(
            "README.md crate entries must be ordered basic user crates first, "
            "then standard extensions, then implementation crates"
        )


def main() -> int:
    errors: list[str] = []
    root_text = (ROOT / "Cargo.toml").read_text(encoding="utf-8")
    metadata = cargo_metadata()
    release_text = (ROOT / "ai/contribution-workflows/release-publish.md").read_text(
        encoding="utf-8"
    )
    check_workspace_members(metadata, root_text, errors)
    check_release_order(metadata, release_text, errors)
    check_workspace_metadata(root_text, errors)
    check_workspace_dependencies(root_text, errors)
    check_package_metadata(root_text, errors)
    check_readme(errors)

    if errors:
        for error in errors:
            print(f"publish-layout: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
