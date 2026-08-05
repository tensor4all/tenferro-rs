#!/usr/bin/env python3
"""Check tenferro publish layout and crates.io-facing metadata.

This check allows pre-publish git dependencies only when they also declare the
registry version Cargo needs for packaging.
"""

from __future__ import annotations

import json
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
    return {
        package["name"]
        for package in metadata["packages"]
        if Path(package["manifest_path"]).parent.parent == ROOT / "crates"
        and package["name"].startswith("tenferro-")
        and package.get("publish") != []
    }


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
    for line_no, line in enumerate(root_text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "git =" in stripped and "version =" not in stripped:
            errors.append(
                f"Cargo.toml:{line_no}: git dependency must include a version "
                "requirement for crates.io packaging"
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
        for line_no, line in enumerate(manifest_text.splitlines(), start=1):
            version_fragment = f'version = "{expected_version}"'
            if 'path = "../tenferro-' in line and version_fragment not in line:
                errors.append(
                    f"{rel(manifest_path)}:{line_no}: tenferro path dependency "
                    f'must include {version_fragment} for crates.io packaging'
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
    check_workspace_members(metadata, root_text, errors)
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
