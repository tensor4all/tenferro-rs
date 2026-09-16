#!/usr/bin/env python3
"""Collect object-level evidence that scalar-set composition shares kernels.

Builds the external proof crate's composition test with `--emit=asm`, then counts
the *defined* instantiations of the shared elementwise kernel and the call sites
that reach them. The claim under test is that one set-specific member does not add
a numerical specialization: the preset instantiation appears once even though two
sets use it, and the external instantiation is the same shared kernel parameterized
by the contribution's own operation type.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_TARGET = "composition"
PACKAGE = "tenferro-df64-proof"
KERNEL = "zip_map2"
# The kernel entry points themselves, as opposed to the threading, dispatch, and
# vectorization wrappers that also carry the kernel name in their symbol.
KERNEL_FUNCTIONS = ("map_view13zip_map2_into", "map_view29zip_map2_parts_into_validated")
WRAPPERS = (
    "rayon_core",
    "threading",
    "ThreadedMapReduce",
    "for_each_inner_block_preordered",
    "join_context",
    "3job",
    "vectorize",
)
EXTERNAL_MEMBER = "tenferro_df64_proof4Df64"
PRESET_OP = "op5AddOp"
EXTERNAL_OP = "7Df64Add"
SHARED_KERNEL_CRATE = "tenferro_internal_cpu_kernels10scalar_ops18scalar_binary_into"

DEFINITION = re.compile(r"^(_R[A-Za-z0-9_]+):$")


def run(*args: str) -> str:
    return subprocess.run(
        args, cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def definitions(assembly: str) -> list[str]:
    """Symbols this assembly file defines."""
    return [match.group(1) for line in assembly.splitlines() if (match := DEFINITION.match(line))]


def references(assembly: str, symbol: str) -> int:
    """How many places use a symbol, excluding its own definition block."""
    pattern = re.compile(rf"(?:\b|,){re.escape(symbol)}(?:\b|,)")
    return sum(
        1
        for line in assembly.splitlines()
        if pattern.search(line)
        and not line.startswith(symbol)
        and "\t.type\t" not in line
        and "\t.size\t" not in line
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    command = [
        "cargo",
        "rustc",
        "-j",
        "16",
        "-p",
        PACKAGE,
        "--test",
        TEST_TARGET,
        "--release",
        "--",
        "--emit=asm",
    ]
    record: dict[str, object] = {
        "schema": "tenferro.scalar-composition-kernel-sharing.v1",
        "candidate_commit": run("git", "rev-parse", "HEAD"),
        "command": " ".join(command),
        "rustc": run("rustc", "-Vv").splitlines()[0],
        "target": run("rustc", "-Vv").split("host: ", 1)[-1].splitlines()[0],
        "status": "inconclusive",
        "observations": [],
    }
    try:
        subprocess.run(command, cwd=ROOT, check=True, text=True)
    except subprocess.CalledProcessError as error:
        record["observations"].append(f"cargo rustc failed with exit code {error.returncode}")
        write_report(args.report, record)
        return 1

    assemblies = sorted(
        (ROOT / "target" / "release" / "deps").glob(f"{TEST_TARGET}-*.s"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not assemblies:
        record["observations"].append("cargo emitted no assembly file")
        write_report(args.report, record)
        return 1

    assembly_path = assemblies[0]
    assembly = assembly_path.read_text(encoding="utf-8", errors="replace")
    defined = definitions(assembly)

    def kernel_entries(member_op: str, external: bool) -> list[str]:
        return sorted(
            {
                name
                for name in defined
                if KERNEL in name
                and any(function in name for function in KERNEL_FUNCTIONS)
                and not any(wrapper in name for wrapper in WRAPPERS)
                and member_op in name
                and ((EXTERNAL_MEMBER in name) == external)
            }
        )

    preset = kernel_entries(PRESET_OP, external=False)
    external = kernel_entries(EXTERNAL_OP, external=True)
    preset_references = sum(references(assembly, symbol) for symbol in preset)
    external_references = sum(references(assembly, symbol) for symbol in external)

    record["assembly"] = str(assembly_path.relative_to(ROOT))
    record["defined_symbols"] = len(defined)
    record["kernel_entries"] = {
        "preset": {
            "count": len(preset),
            "references": preset_references,
            "by_function": {
                function: [symbol for symbol in preset if function in symbol]
                for function in KERNEL_FUNCTIONS
            },
            "all_from_shared_kernel_crate": bool(preset)
            and all(SHARED_KERNEL_CRATE in symbol for symbol in preset),
        },
        "external": {
            "count": len(external),
            "references": external_references,
            "by_function": {
                function: [symbol for symbol in external if function in symbol]
                for function in KERNEL_FUNCTIONS
            },
            "all_from_shared_kernel_crate": bool(external)
            and all(SHARED_KERNEL_CRATE in symbol for symbol in external),
            "all_parameterized_by_contribution_op": bool(external)
            and all(EXTERNAL_OP in symbol for symbol in external),
        },
    }

    failures: list[str] = []
    for function in KERNEL_FUNCTIONS:
        if not record["kernel_entries"]["preset"]["by_function"][function]:
            failures.append(f"the preset path instantiates no {function}")
        if not record["kernel_entries"]["external"]["by_function"][function]:
            failures.append(f"the external path instantiates no {function}")
    if not record["kernel_entries"]["preset"]["all_from_shared_kernel_crate"]:
        failures.append("a preset instantiation does not come from the shared kernel crate")
    if not record["kernel_entries"]["external"]["all_from_shared_kernel_crate"]:
        failures.append("an external instantiation does not come from the shared kernel crate")
    if not record["kernel_entries"]["external"]["all_parameterized_by_contribution_op"]:
        failures.append(
            "an external instantiation is not parameterized by the contribution's operation"
        )
    if preset_references < 2:
        failures.append(
            f"the preset instantiations are reached from {preset_references} places, "
            "so they are not visibly shared between call sites"
        )

    record["failures"] = failures
    record["status"] = "fail" if failures else "pass"
    write_report(args.report, record)
    print(json.dumps(record, indent=2))
    return 1 if failures else 0


def write_report(path: Path, record: dict[str, object]) -> None:
    path.write_text(
        "# Scalar composition kernel sharing\n\n"
        "Generated by `scripts/check-scalar-composition-kernel-sharing.py`. The record\n"
        "below is the object-level evidence for the claim that composing scalar sets\n"
        "shares one compiled numerical kernel per element type and operation rather than\n"
        "adding a set-specific specialization.\n\n"
        "```json\n" + json.dumps(record, indent=2) + "\n```\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
