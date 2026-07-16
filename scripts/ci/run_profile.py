#!/usr/bin/env python3
"""Run exact command profiles shared by contributors and hosted CI."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Sequence
from typing import TextIO


PROFILE_COMMANDS: dict[str, tuple[str, ...]] = {
    "local-gate": (
        "cargo nextest run --workspace --cargo-profile local-gate "
        "--no-fail-fast",
        "cargo test --doc --workspace --profile local-gate",
    ),
    "workspace-faer": (
        "cargo nextest run --workspace --release --no-fail-fast",
        "cargo test --doc --workspace --release",
    ),
    "workspace-blas": (
        "cargo nextest run --workspace --release --no-default-features "
        "--features cpu-blas --no-fail-fast",
        "cargo test --doc --workspace --release --no-default-features "
        "--features cpu-blas",
    ),
    "blas-inject": (
        "cargo test -p tenferro-cpu --test inject_tests --release "
        '--no-default-features --features "cpu-blas,provider-inject"',
    ),
    "extensions": (
        "cargo test --manifest-path ext/tropical/Cargo.toml --release "
        "--features autodiff",
        "cargo test --manifest-path ext/sparse/Cargo.toml --release "
        "--features autodiff",
        "cargo check --manifest-path samples/kdv-pinn/Cargo.toml --release "
        "--all-targets",
    ),
    "docs": (
        "python3 scripts/test-check-docs-site.py",
        "python3 scripts/test-doc-consistency.py",
        "python3 scripts/test-repository-rules-review.py",
        "python3 scripts/check-guide-dependency-snippets.py",
        "python3 scripts/check-operation-categories.py --fail-on-findings",
        "bash scripts/build_docs_site.sh",
    ),
    "coverage": (
        "cargo llvm-cov --workspace --release --json --output-path coverage.json",
        "python3 scripts/check-coverage.py coverage.json",
    ),
    "ci-config": (
        "python3 -m unittest discover -s scripts/ci/tests -v",
        "actionlint",
    ),
}

FULL_PROFILE = (
    "workspace-faer",
    "workspace-blas",
    "blas-inject",
    "extensions",
    "docs",
    "coverage",
    "ci-config",
)


def commands_for(profile: str) -> tuple[str, ...]:
    """Return immutable commands for one concrete profile."""

    try:
        return PROFILE_COMMANDS[profile]
    except KeyError as error:
        raise ValueError(f"unknown CI profile: {profile}") from error


def expand_profiles(profiles: Sequence[str]) -> tuple[str, ...]:
    """Expand composites and preserve the first occurrence of each profile."""

    expanded: list[str] = []
    for profile in profiles:
        names = FULL_PROFILE if profile == "full" else (profile,)
        for name in names:
            commands_for(name)
            if name not in expanded:
                expanded.append(name)
    return tuple(expanded)


def run_profiles(
    profiles: Sequence[str], *, dry_run: bool, output: TextIO = sys.stdout
) -> None:
    """Run selected profiles in order, or print their commands in dry-run mode."""

    for profile in expand_profiles(profiles):
        for command in commands_for(profile):
            print(f"+ {command}", file=output, flush=True)
            if dry_run:
                continue
            environment = os.environ.copy()
            if profile == "workspace-blas":
                environment["RUSTFLAGS"] = "-l dylib=openblas -l dylib=lapack"
            try:
                # Commands are repository constants, not caller-provided shell text.
                subprocess.run(
                    command, shell=True, check=True, env=environment
                )
            except subprocess.CalledProcessError as error:
                raise RuntimeError(
                    f"CI profile {profile!r} failed: {command}"
                ) from error


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profiles", nargs="*")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if not args.list and not args.profiles:
        parser.error("provide at least one profile or --list")
    return args


def main() -> int:
    args = _parse_args()
    if args.list:
        print("full: " + ", ".join(FULL_PROFILE))
        for name in PROFILE_COMMANDS:
            print(name)
        if not args.profiles:
            return 0
    try:
        run_profiles(args.profiles, dry_run=args.dry_run)
    except (RuntimeError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
