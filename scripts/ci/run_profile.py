#!/usr/bin/env python3
"""Run exact command profiles shared by contributors and hosted CI."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

# Hosted CI uses Cargo `[profile.ci]`: opt-level=0, debug=0, incremental=false,
# strip="symbols". nextest takes `--cargo-profile`; cargo/llvm-cov take `--profile`.
_CARGO_PROFILE = "ci"
_NEXTEST_PROFILE = f"--cargo-profile {_CARGO_PROFILE}"
_CARGO_TEST_PROFILE = f"--profile {_CARGO_PROFILE}"
_CLIPPY_FLAGS = (
    "-D warnings -D clippy::missing_errors_doc -D clippy::missing_panics_doc"
)
_STORAGE_OWNERSHIP_CHECKER = (
    "python3 scripts/check-storage-ownership-contracts.py"
)
_TRYBUILD_CARGO_WRAPPER = str(Path(__file__).with_name("trybuild-cargo.py").resolve())

PROFILE_COMMANDS: dict[str, tuple[str, ...]] = {
    "fmt": (
        "cargo fmt --all --check",
        "cargo fmt --manifest-path ext/tropical/Cargo.toml --all --check",
        "cargo fmt --manifest-path ext/sparse/Cargo.toml --all --check",
        "cargo fmt --manifest-path ext/tenferro-cpu-tblis/Cargo.toml --all --check",
    ),
    "clippy": (
        f"cargo clippy --workspace --all-targets -- {_CLIPPY_FLAGS}",
        "cargo clippy --manifest-path ext/tropical/Cargo.toml --all-targets -- "
        f"{_CLIPPY_FLAGS}",
        "cargo clippy --manifest-path ext/sparse/Cargo.toml --all-targets -- "
        f"{_CLIPPY_FLAGS}",
        "cargo clippy --manifest-path ext/tenferro-cpu-tblis/Cargo.toml --all-targets -- "
        f"{_CLIPPY_FLAGS}",
    ),
    "workspace-faer": (
        f"cargo nextest run --workspace {_NEXTEST_PROFILE} --no-fail-fast",
        f"cargo test --doc --workspace {_CARGO_TEST_PROFILE}",
    ),
    "workspace-blas": (
        # Direct invocation preserves our CARGO wrapper in nextest test processes.
        f"cargo-nextest nextest run --workspace {_NEXTEST_PROFILE} --no-default-features "
        "--features cpu-blas --no-fail-fast",
        f"cargo test --doc --workspace {_CARGO_TEST_PROFILE} --no-default-features "
        "--features cpu-blas",
        # Downstream BLAS interop example (issue #1602): links the system
        # OpenBLAS/LAPACK through the profile RUSTFLAGS, so native symbol
        # linkage is verified, not just compilation.
        f"cargo run -p tenferro-tutorial-code {_CARGO_TEST_PROFILE} --no-default-features "
        "--features cpu-blas --bin blas_interop",
    ),
    "blas-inject": (
        f"cargo test -p tenferro-cpu {_CARGO_TEST_PROFILE} --no-default-features "
        '--features "cpu-blas,provider-inject" --test integration inject_tests',
    ),
    "extensions": (
        f"cargo test --manifest-path ext/tropical/Cargo.toml {_CARGO_TEST_PROFILE} "
        "--features autodiff",
        f"cargo test --manifest-path ext/sparse/Cargo.toml {_CARGO_TEST_PROFILE} "
        "--features autodiff",
        f"cargo test --manifest-path ext/tenferro-cpu-tblis/Cargo.toml {_CARGO_TEST_PROFILE}",
        f"cargo check --manifest-path samples/kdv-pinn/Cargo.toml {_CARGO_TEST_PROFILE} "
        "--all-targets",
        f"cargo test --manifest-path samples/cubecl-kernel/Cargo.toml {_CARGO_TEST_PROFILE} "
        "--all-targets",
    ),
    "docs": (
        "python3 scripts/test-check-docs-site.py",
        "python3 scripts/test-gen-dep-graph.py",
        "python3 scripts/test-doc-consistency.py",
        "python3 scripts/test-repository-rules-review.py",
        "python3 scripts/test-check-guide-dependency-snippets.py",
        "python3 scripts/check-guide-dependency-snippets.py",
        "python3 scripts/check-operation-categories.py --fail-on-findings",
        # Downstream external-linalg interop examples (issue #1602): compiled
        # and run as a consumer that uses only public tenferro APIs. The BLAS
        # binary links the system OpenBLAS/LAPACK installed by the docs CI job
        # (the RUSTFLAGS are scoped to this one command).
        f"cargo run -p tenferro-tutorial-code {_CARGO_TEST_PROFILE} --no-default-features "
        "--features cpu-faer --bin faer_interop",
        f"RUSTFLAGS='-l dylib=openblas -l dylib=lapack' cargo run -p tenferro-tutorial-code "
        f"{_CARGO_TEST_PROFILE} --no-default-features --features cpu-blas --bin blas_interop",
        # Issue #1724: compile the source-backed raw CUDA examples on GPU-less
        # docs CI; hardware execution remains in the CUDA test lane.
        f"cargo check -p tenferro-tutorial-code {_CARGO_TEST_PROFILE} --no-default-features "
        "--features cuda,cpu-faer --bin custom_cuda_kernels",
        "bash scripts/build_docs_site.sh",
    ),
    "coverage": (
        f"cargo llvm-cov --workspace --exclude tenferro-tutorial-code "
        f"{_CARGO_TEST_PROFILE} --json --output-path coverage.json",
        "python3 scripts/check-coverage.py coverage.json",
    ),
    "ci-config": (
        "python3 scripts/test-release-publish.py",
        "python3 scripts/test-check-publish-layout.py",
        "python3 scripts/test-release-validation-policy.py",
        "python3 scripts/check-publish-layout.py",
        "python3 scripts/test-storage-ownership-contracts-v2.py",
        _STORAGE_OWNERSHIP_CHECKER,
        "python3 -m unittest discover -s scripts/ci/tests -v",
        "actionlint",
    ),
}

FULL_PROFILE = (
    "fmt",
    "clippy",
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


def _command_for_execution(command: str) -> str:
    """Use this process's interpreter for repository-owned Python commands."""

    if command == "python3" or command.startswith("python3 "):
        return shlex.quote(sys.executable) + command[len("python3") :]
    return command


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
    profiles: Sequence[str],
    *,
    dry_run: bool,
    output: TextIO = sys.stdout,
    storage_ownership_base: str | None = None,
) -> None:
    """Run selected profiles in order, or print their commands in dry-run mode."""

    expanded_profiles = expand_profiles(profiles)
    if storage_ownership_base is not None and "ci-config" not in expanded_profiles:
        raise ValueError(
            "storage ownership base requires the ci-config profile"
        )

    for profile in expanded_profiles:
        for command in commands_for(profile):
            if (
                storage_ownership_base is not None
                and profile == "ci-config"
                and command == _STORAGE_OWNERSHIP_CHECKER
            ):
                command += (
                    " --base-commit " + shlex.quote(storage_ownership_base)
                )
            print(f"+ {command}", file=output, flush=True)
            if dry_run:
                continue
            environment = os.environ.copy()
            environment["PYTHON"] = sys.executable
            if profile == "workspace-blas":
                rustflags = "-l dylib=openblas -l dylib=lapack"
                environment["RUSTFLAGS"] = rustflags
                environment["TENFERRO_TRYBUILD_RUSTFLAGS"] = rustflags
                environment["CARGO"] = _TRYBUILD_CARGO_WRAPPER
            execution_command = _command_for_execution(command)
            try:
                # Commands are repository constants, not caller-provided shell text.
                subprocess.run(
                    execution_command, shell=True, check=True, env=environment
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
    parser.add_argument("--storage-ownership-base")
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
        run_profiles(
            args.profiles,
            dry_run=args.dry_run,
            storage_ownership_base=args.storage_ownership_base,
        )
    except (RuntimeError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
