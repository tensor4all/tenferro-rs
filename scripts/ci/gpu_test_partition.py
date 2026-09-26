#!/usr/bin/env python3
"""Run an audited, exhaustive host/device partition of a nextest archive."""

from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path
import subprocess
import tempfile

LANES = {"host", "gpu", "two-gpu", "compile-only", "benchmark", "external-tool"}


def read_partition(path: Path) -> dict[tuple[str, str], str]:
    result = {}
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line or line.startswith("#"):
            continue
        lane, binary, name = line.split("\t")
        key = (binary, name)
        if lane not in LANES or key in result:
            raise ValueError(f"{path}:{number}: invalid lane or duplicate test: {line}")
        result[key] = lane
    return result


def validate_inventory(inventory: dict, partition: dict[tuple[str, str], str]) -> None:
    actual = {
        (binary, name)
        for binary, suite in inventory["rust-suites"].items()
        for name in suite["testcases"]
    }
    expected = set(partition)
    if actual != expected:
        added = sorted(actual - expected)
        removed = sorted(expected - actual)
        raise ValueError(
            f"GPU test inventory changed; audit test bodies/helpers and update the partition. "
            f"Unclassified: {added}; absent: {removed}"
        )
    if not actual or len(actual) != inventory["test-count"]:
        raise ValueError("empty or inconsistent nextest inventory")


def filter_expression(partition: dict[tuple[str, str], str], lane: str) -> str:
    # The inventory was checked first: the complement is exhaustive, not a
    # default classification for unknown/new tests. Keep host argv bounded.
    selected = collections.defaultdict(list)
    for (binary, name), assigned in sorted(partition.items()):
        include = assigned != "host" if lane == "host" else assigned == lane
        if include:
            selected[binary].append(f"test(={name})")
    terms = []
    for binary, tests in selected.items():
        package, _, target = binary.partition("::")
        target = target or package.replace("-", "_")
        terms.append(
            f"(package(={package}) & binary(={target}) & ({' | '.join(tests)}))"
        )
    expression = " | ".join(terms) or "none()"
    return f"not ({expression})" if lane == "host" else expression


def report(partition: dict[tuple[str, str], str], lane: str) -> str:
    counts = collections.Counter(partition.values())
    lines = [f"GPU archive partition: {dict(sorted(counts.items()))}; executing {lane}."]
    for (binary, name), assigned in sorted(partition.items()):
        if assigned not in {"host", "gpu"}:
            lines.append(f"NOT RUN ({assigned}): {binary} {name}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", required=True, choices=("cuda", "pjrt"))
    parser.add_argument("--lane", required=True, choices=("host", "gpu"))
    parser.add_argument("--archive-file", required=True, type=Path)
    parser.add_argument("--workspace-remap", type=Path, default=Path.cwd())
    args = parser.parse_args()
    partition = read_partition(Path(__file__).with_name(f"{args.kind}_test_partition.tsv"))
    with tempfile.TemporaryDirectory(prefix="tenferro-tests-") as temporary:
        root = Path(temporary)
        subprocess.run([
            "cargo", "nextest", "list", "--archive-file", str(args.archive_file),
            "--extract-to", temporary, "--workspace-remap", str(args.workspace_remap),
            "--list-type", "binaries-only", "--message-format", "json",
        ], check=True, stdout=subprocess.DEVNULL)
        # Reuse the extraction for listing and execution; never rebuild on RunPod.
        reuse = [
            "--cargo-metadata", str(root / "target/nextest/cargo-metadata.json"),
            "--binaries-metadata", str(root / "target/nextest/binaries-metadata.json"),
            "--target-dir-remap", str(root / "target"),
            "--workspace-remap", str(args.workspace_remap),
        ]
        listed = subprocess.run([
            "cargo", "nextest", "list", *reuse, "--message-format", "json",
        ], check=True, capture_output=True, text=True)
        validate_inventory(json.loads(listed.stdout), partition)
        summary = report(partition, args.lane)
        print(summary, flush=True)
        if summary_path := os.environ.get("GITHUB_STEP_SUMMARY"):
            with open(summary_path, "a") as stream:
                stream.write(f"```text\n{summary}```\n")
        env = os.environ.copy()
        timeout_profile = []
        if args.lane == "gpu":
            env["TENFERRO_REQUIRE_GPU"] = "1"
            timeout_profile = [
                "--config-file", str(Path(__file__).with_name("gpu_nextest.toml")),
                "--profile", "gpu-ci",
            ]
        subprocess.run([
            "cargo", "nextest", "run", *timeout_profile, *reuse, "--run-ignored", "all",
            "--no-fail-fast", "-j", "1", "-E", filter_expression(partition, args.lane),
        ], check=True, env=env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
