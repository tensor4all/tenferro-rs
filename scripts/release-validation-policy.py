#!/usr/bin/env python3
"""Classify the validation lane required for a release-time change set.

Release publication is irreversible, so the maintainer must re-validate the
tagged tree before publishing. Not every change needs the full workspace and
CPU/GPU suites: a version-bump-only diff cannot change kernel behavior, and a
change confined to the release helpers cannot change crate code at all. This
classifier assigns each changed path to the strongest lane that any change
demands:

- ``helper-or-workflow-only``: changes confined to the release helper,
  publish-layout checker, their tests, the CI profile runner, the canonical
  release workflow, the release skill adapters, or the PR CI workflow
  definition. Run the focused ``ci-config`` lane (python release-helper tests,
  publish-layout check, ``bash -n``, non-uploading dry-run reproductions).
  No full workspace build, no CPU/GPU suites.
- ``publication-metadata-only``: changes to Cargo metadata fields only
  (``version``, ``description``, ``homepage``, ``keywords``, ``categories``,
  ``documentation``, ``readme``, in ``[package]`` or ``[workspace.package]``).
  Run metadata, publish-layout, and archive/dry-run checks only. No CPU/GPU or
  full workspace rerun when exact-SHA CI already passed for the commit.
- ``semantic-manifest``: any other Cargo.toml change (dependency sources,
  versions, or features, target config, ``build.rs``, native libraries,
  profiles, added/removed manifests). Run affected tests plus the applicable
  CI tier.
- ``full``: Rust source changes and anything unclassifiable. Run the full
  normal validation; conservative by default.

Mixed change sets select the strongest (most validating) lane.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tomllib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import NamedTuple


LANE_HELPER = "helper-or-workflow-only"
LANE_METADATA = "publication-metadata-only"
LANE_SEMANTIC = "semantic-manifest"
LANE_FULL = "full"

LANE_STRENGTH = {LANE_HELPER: 0, LANE_METADATA: 1, LANE_SEMANTIC: 2, LANE_FULL: 3}

ROOT = Path(__file__).resolve().parents[1]

# Release machinery whose own tests cover it; changes here never alter crate
# code, so the focused ci-config lane is sufficient. Only the release-related
# test files that the ci-config profile actually executes are helper-only;
# any other scripts/test-*.py change is classified conservatively as full.
RELEASE_HELPER_PATHS = frozenset(
    {
        "scripts/release-publish.py",
        "scripts/check-publish-layout.py",
        "scripts/release-validation-policy.py",
        "scripts/ci/run_profile.py",
        "scripts/test-release-publish.py",
        "scripts/test-check-publish-layout.py",
        "scripts/test-release-validation-policy.py",
        "ai/contribution-workflows/release-publish.md",
        ".agents/skills/tenferro-release-publish/SKILL.md",
        ".claude/skills/tenferro-release-publish/SKILL.md",
        ".kimi/skills/tenferro-release-publish/SKILL.md",
        ".opencode/commands/tenferro-release-publish.md",
        ".github/workflows/ci-pr-workspace-tests.yml",
    }
)

# Cargo [package] fields that describe publication metadata only. A change to
# any of them (in [package] or [workspace.package]) cannot alter crate code,
# dependency resolution, or build behavior.
MANIFEST_METADATA_FIELDS = frozenset(
    {
        "version",
        "description",
        "homepage",
        "keywords",
        "categories",
        "documentation",
        "readme",
    }
)


class ManifestFields(NamedTuple):
    metadata: dict[str, object]
    structural: dict[str, object]


def _manifest_fields(text: bytes) -> ManifestFields:
    """Split one manifest into publication-metadata and structural fields."""

    parsed = tomllib.loads(text.decode("utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError("Cargo.toml root must be a table")
    metadata: dict[str, object] = {}
    structural: dict[str, object] = {}
    package = parsed.get("package")
    if isinstance(package, dict):
        for key, value in package.items():
            if key in MANIFEST_METADATA_FIELDS:
                metadata[key] = value
            else:
                structural[key] = value
    workspace = parsed.get("workspace")
    if isinstance(workspace, dict):
        workspace_package = workspace.get("package")
        if isinstance(workspace_package, dict):
            for key, value in workspace_package.items():
                if key in MANIFEST_METADATA_FIELDS:
                    metadata[f"workspace.package.{key}"] = value
                else:
                    structural[f"workspace.package.{key}"] = value
        for key, value in workspace.items():
            if key != "package":
                structural[f"workspace.{key}"] = value
    for key, value in parsed.items():
        if key not in ("package", "workspace"):
            structural[key] = value
    return ManifestFields(metadata, structural)


def classify_change(path: str, old: bytes | None, new: bytes | None) -> str:
    """Return the validation lane for one changed path."""

    if path in RELEASE_HELPER_PATHS:
        return LANE_HELPER
    if path.endswith("Cargo.toml"):
        try:
            old_fields = _manifest_fields(old) if old is not None else None
            new_fields = _manifest_fields(new) if new is not None else None
        except (tomllib.TOMLDecodeError, ValueError, UnicodeDecodeError):
            return LANE_SEMANTIC
        if old_fields is None or new_fields is None:
            return LANE_SEMANTIC
        if old_fields.structural == new_fields.structural:
            return LANE_METADATA
        return LANE_SEMANTIC
    # Rust sources, non-manifest config, docs, and anything unrecognized:
    # conservative full validation.
    return LANE_FULL


def classify_changes(changes: Sequence[tuple[str, bytes | None, bytes | None]]) -> str:
    """Return the strongest lane demanded by any change in the set."""

    strongest = LANE_HELPER
    for path, old, new in changes:
        lane = classify_change(path, old, new)
        if LANE_STRENGTH[lane] > LANE_STRENGTH[strongest]:
            strongest = lane
    return strongest


def _git_show(
    revision: str, path: str, runner: Callable
) -> bytes | None:
    completed = runner(["git", "show", f"{revision}:{path}"], check=False)
    return completed.stdout if completed.returncode == 0 else None


def classify_diff(
    base: str,
    head: str,
    *,
    runner: Callable = subprocess.run,
    root: Path = ROOT,
) -> str:
    """Classify every path changed between two git revisions."""

    completed = runner(
        ["git", "diff", "--name-status", base, head],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    changes: list[tuple[str, bytes | None, bytes | None]] = []
    for line in completed.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) == 2:
            status, path = parts
        elif len(parts) == 3:
            _old_path, status, path = parts  # rename/copy pair
        else:
            continue
        old = None if status.startswith("A") else _git_show(base, path, runner)
        new = None if status.startswith("D") else _git_show(head, path, runner)
        changes.append((path, old, new))
    return classify_changes(changes)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--base",
        metavar="SHA",
        help="classify all paths changed between --base and --head in the repo",
    )
    source.add_argument(
        "--change",
        nargs=3,
        action="append",
        metavar=("PATH", "OLD", "NEW"),
        help="classify one path with old and new content files ('-' when absent); "
        "repeatable",
    )
    parser.add_argument("--head", metavar="SHA", help="head revision for --base")
    args = parser.parse_args(argv)

    if args.base is not None:
        if args.head is None:
            parser.error("--base requires --head")
        lane = classify_diff(args.base, args.head)
    else:
        changes: list[tuple[str, bytes | None, bytes | None]] = []
        for path, old_name, new_name in args.change:
            try:
                old = None if old_name == "-" else Path(old_name).read_bytes()
                new = None if new_name == "-" else Path(new_name).read_bytes()
            except OSError as error:
                parser.error(f"could not read change content for {path}: {error}")
            changes.append((path, old, new))
        lane = classify_changes(changes)
    print(lane)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
