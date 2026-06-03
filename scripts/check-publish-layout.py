#!/usr/bin/env python3
"""Check tenferro publish layout and crates.io-facing metadata.

This check intentionally does not reject git dependencies. Converting those
dependencies to registry versions is a separate release-prep step.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
USER_CRATE_ORDER = [
    "tenferro-tensor",
    "tenferro-cpu",
    "tenferro-gpu",
    "tenferro-runtime",
    "tenferro-ad",
]
EXTENSION_CRATE_ORDER = [
    "tenferro-linalg",
    "tenferro-einsum",
    "tenferro-fft",
]
IMPLEMENTATION_CRATE_ORDER = [
    "tenferro-tensor-core",
    "tenferro-core-ops",
    "tenferro-internal-ops",
    "tenferro-internal-extension-macros",
]
TENFERRO_CRATES = (
    USER_CRATE_ORDER + EXTENSION_CRATE_ORDER + IMPLEMENTATION_CRATE_ORDER
)


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


def check_workspace_members(metadata: dict, root_text: str, errors: list[str]) -> None:
    packages = {package["name"]: package for package in metadata["packages"]}

    if "tenferro" in packages:
        errors.append("workspace must not include a root tenferro facade crate")

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
        'publish = true',
        'readme = "README.md"',
        'repository = "https://github.com/tensor4all/tenferro-rs"',
        'homepage = "https://tensor4all.org/tenferro-rs/"',
    ]
    for line in required_lines:
        if line not in package_section:
            errors.append(f"workspace.package missing {line!r}")


def check_package_metadata(errors: list[str]) -> None:
    for crate in TENFERRO_CRATES:
        manifest_path = ROOT / "crates" / crate / "Cargo.toml"
        if not manifest_path.exists():
            errors.append(f"missing manifest {rel(manifest_path)}")
            continue

        manifest_text = manifest_path.read_text(encoding="utf-8")
        package_section = section(manifest_text, "package")
        for key in ("publish", "readme", "repository", "homepage"):
            if f"{key}.workspace = true" not in package_section:
                errors.append(
                    f"{rel(manifest_path)} package.{key} must inherit workspace metadata"
                )
        for line_no, line in enumerate(manifest_text.splitlines(), start=1):
            if 'path = "../tenferro-' in line and 'version = "0.1.0"' not in line:
                errors.append(
                    f"{rel(manifest_path)}:{line_no}: tenferro path dependency "
                    'must include version = "0.1.0" for crates.io packaging'
                )

    tutorial_manifest = ROOT / "docs" / "tutorial-code" / "Cargo.toml"
    if "publish = false" not in section(
        tutorial_manifest.read_text(encoding="utf-8"), "package"
    ):
        errors.append("docs/tutorial-code must remain publish = false")

    tropical_manifest = ROOT / "ext" / "tropical" / "Cargo.toml"
    if "publish = false" not in section(
        tropical_manifest.read_text(encoding="utf-8"), "package"
    ):
        errors.append("ext/tropical must remain publish = false")


def check_readme(errors: list[str]) -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    required_fragments = [
        "There is intentionally no `tenferro` facade crate",
        "### Core User Crates",
        "### Standard Operation Extensions",
        "### Published Implementation Crates",
    ]
    for fragment in required_fragments:
        if fragment not in readme:
            errors.append(f"README.md missing fragment: {fragment!r}")

    ordered = USER_CRATE_ORDER + EXTENSION_CRATE_ORDER + IMPLEMENTATION_CRATE_ORDER
    positions = []
    for crate in ordered:
        marker = f"`{crate}`"
        pos = readme.find(marker)
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
    check_package_metadata(errors)
    check_readme(errors)

    if errors:
        for error in errors:
            print(f"publish-layout: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
